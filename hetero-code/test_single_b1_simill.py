import os
import time
import functorch.compile
import functorch.compile
import torch
import torch._dynamo.config
import deepspeed
from deepspeed.accelerator import get_accelerator
from transformers import AutoTokenizer, LlamaForCausalLM, DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoConfig, LlamaConfig
from datasets import load_dataset
import datasets
from huggingface_hub import login
from torch.utils.data import DataLoader
import logging
import torch.cuda.nvtx as nvtx
import argparse
import itertools

# ----------------------------
# 데이터 준비 함수
# ----------------------------
# max_length길이만큼 연속 청크로 묶고, 필요 시 split_size만큼 샘플을 선택
# pad 토큰은 라벨에서 -100으로 마스킹하여 loss에서 제외
def load_wikitext(tokenizer, collator, max_length=None, split_size=None):
    ds = datasets.load_dataset("wikitext", "wikitext-2-v1")["train"]

    # 1) 토크나이즈 (패딩/트렁케이션 하지 않음; special tokens은 수동으로 EOS만 삽입)
    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return out

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"]) 

    # 2) 연속 청크로 묶기 (문서 사이에 EOS 삽입)
    def group_texts(examples):
        eos_id = tokenizer.eos_token_id
        # 모든 시퀀스를 하나로 이어 붙이되, 각 시퀀스 사이에 EOS를 삽입
        all_ids = []
        for ids in examples["input_ids"]:
            all_ids.extend(ids + [eos_id])
        # block 단위로 자르기
        total_len = (len(all_ids) // max_length) * max_length
        all_ids = all_ids[:total_len]
        input_ids = [all_ids[i : i + max_length] for i in range(0, total_len, max_length)]
        labels = [ids.copy() for ids in input_ids]
        # 3) pad 마스킹 (안전장치: pad가 존재하면 -100)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
        if pad_id != -1:
            for row in labels:
                for j in range(len(row)):
                    if row[j] == pad_id:
                        row[j] = -100
        return {"input_ids": input_ids, "labels": labels}

    ds = ds.map(group_texts, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "labels"]) 

    if split_size is not None:
        print(f"[Datasets] len (grouped blocks): {len(ds)}")
        ds = ds.select(range(min(split_size, len(ds))))
    return ds

# ----------------------------
# 메인
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_nsys", "-n", type=int, default=0,
                        help="if set to 1, it will be profiled with nsys including nvtx (default=0)")
    parser.add_argument("--log_file", "-f", type=str, default="training.log",
                        help="로그 파일명 (default=train.log)")
    parser.add_argument("--model", "-m", type=str, default="llama",
                        help="model name. (default=Llama)(option: llama, opt)")
    parser.add_argument("--max_steps", "-s", type=int, default=1,
                        help="최대 배치 수 (default=1)")
    parser.add_argument("--max_length", "-l", type=int, default=512,
                        help="sample length (default=1024)")
    parser.add_argument("--checkpoint", "-c", type=int, default=0,
                        help="if set to 0, will not checkpoint.(default=0)")
    parser.add_argument("--local_rank", "-rank", type=int, default=0,
                        help="local rank (default=0)")
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # ----------------------------
    # 로그 설정
    # ----------------------------
    logging.basicConfig(
        filename="./training.log",
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("deepspeed")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("/home/deepspeed/hetero-code/"+args.log_file, mode="w")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                    datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    logger.handlers = [file_handler]
    logger.info("Logger initialized.")

    max_step = args.max_steps
    # single-node 환경: local_rank 로 GPU 지정
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logger.info(f"Using device {device}")
    
    # 로그인, 토크나이저, collator
    hug_token = os.environ["HUG_TOKEN"]
    login(hug_token)
    model_name = "meta-llama/Llama-3.2-1B"
    opt_model = "facebook/opt-125m"
    if args.model == "opt":
        tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
    elif args.model == "llama":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # 데이터 로드 & 전처리 (예제에선 64개만 사용)
    logger.info("Loading and tokenizing dataset...")
    train_dataset = load_wikitext(tokenizer, collator, max_length=args.max_length, split_size=16000)
    
    # DeepSpeed 설정
    ds_config = "/home/deepspeed/hetero-code/config/ds_config_stage2.json"
    logger.info("Loading model under ZeRO init...")
    if args.use_nsys == 1 and local_rank == 0:
        torch.cuda.cudart().cudaProfilerStart()
        nvtx.range_push("train")
    # with deepspeed.zero.Init(remote_device="nvme", config_dict_or_path=ds_config):
    if args.model == "opt":
        cfg = AutoConfig.from_pretrained(opt_model)
        model = AutoModelForCausalLM.from_config(cfg).to(device).train()
    elif args.model == "llama":
        cfg = LlamaConfig.from_pretrained(model_name)
        model = LlamaForCausalLM(cfg).to(device).train()

    # ensure embedding size matches tokenizer size (important when training from scratch)
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # set AC no-complie
    # model.gradient_checkpointing_enable() 

    # 엔진 초기화 
    logger.info("Initializing DeepSpeed engine...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        # training_data=train_dataset
    )
    logger.info("DeepSpeed engine initialized.")

    # DataLoader (단일 노드/단일 프로세스)
    batch_size = model_engine.train_micro_batch_size_per_gpu()
    logger.info(f"Batch size: {batch_size}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1
    )

    # model_engine.compile()
    # model_engine = torch.compile(model_engine)
    
    # 학습 루프
    training_time = 0.0
    loss = None
    data_iter = iter(train_loader)
    step = 0
    logger.info(f"Target max_steps: {max_step}")
    while step < max_step:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        st = time.time()
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        labels   = batch["labels"].to(device)

        # (3) pad 무시 비율 로깅: labels==-100 은 loss에서 제외되는 토큰
        try:
            masked_ratio = (labels == -100).float().mean().item()
            logger.info(f"Masked(label=-100) ratio: {masked_ratio:.2%}")
        except Exception:
            pass
        
        outputs = model_engine(input_ids=input_ids, labels=labels)

        loss = outputs.loss
        model_engine.backward(loss)

        model_engine.step()
        step += 1

        logger.info(f"Step {step} Loss {loss.item():.4f}")
        torch.cuda.synchronize()

        ed = time.time() - st
        training_time+=ed

        logger.info(f"[MAIN] Step Time: {ed:.4f}s")
    
    client_sd = {}
    if args.checkpoint == 1:
        client_sd['step'] =int(step+1)
        chpt_id = int(step+1)
        model_engine.save_checkpoint(save_dir = "./checkpoint", tag = chpt_id, client_state = client_sd, save_latest = True)

    logger.info(f"training_time: {training_time:.4f}s")
    if args.use_nsys == 1 and local_rank == 0:
        nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

    logger.info("Training finished (checkpoint skipped).")

if __name__ == "__main__":
    main()
