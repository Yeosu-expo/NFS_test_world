import os
import time
import functorch.compile
import functorch.compile
import torch
import torch._dynamo.config
import deepspeed
from deepspeed.accelerator import get_accelerator
from transformers import AutoTokenizer, LlamaForCausalLM, DataCollatorForLanguageModeling, AutoModelForCausalLM
from datasets import load_dataset
import datasets
from huggingface_hub import login
from torch.utils.data import DataLoader
import logging
import torch.cuda.nvtx as nvtx

# ----------------------------
# 로그 설정
# ----------------------------
logging.basicConfig(
    filename="/root/training.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("deepspeed")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("/root/deepspeed_inter.log", mode="w")
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                   datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)
logger.handlers = [file_handler]
logger.info("Logger initialized.")

# ----------------------------
# 데이터 준비 함수
# ----------------------------
# max_length길이만큼 데이터 길이를 맞추고, 전체 데이터 중에 split_size만큼의 샘플을 반환함
# batch_size가 4이고 split_size가 64라면, 각 스텝에서 4개의 샘플을 수행하고 이를 총 16 스텝 수행하여, 모든 샘플을 처리하게됨
def load_wikitext(tokenizer, collator, max_length=None, split_size=None):
    def mask_tokens(x):
        input_ids, labels = collator.torch_mask_tokens(
            x['input_ids'], special_tokens_mask=x['special_tokens_mask']
        )
        return {"input_ids": input_ids, "labels": labels}

    ds = datasets.load_dataset("wikitext", "wikitext-2-v1")["train"]
    # tokenization
    ds = ds.map(lambda x: tokenizer(
                        x["text"],
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                        return_special_tokens_mask=True),
                batched=True)
    ds.set_format(type="torch", columns=["input_ids", "special_tokens_mask"])
    if collator.mlm:
        ds = ds.map(mask_tokens, remove_columns=["special_tokens_mask"])
    else:
        ds = ds.map(lambda x: {"input_ids": x["input_ids"], "labels": x["input_ids"]})
    if split_size is not None:
        ds = ds.select(range(split_size))
    return ds

# ----------------------------
# 메인
# ----------------------------
def main():
    max_step = 4
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
    tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # 데이터 로드 & 전처리 (예제에선 64개만 사용)
    logger.info("Loading and tokenizing dataset...")
    train_dataset = load_wikitext(tokenizer, collator, max_length=512, split_size=64)
    
    # DeepSpeed 설정
    ds_config = "/home/deepspeed/config/ds_config_stage2.json"
    logger.info("Loading model under ZeRO init...")
    if local_rank == 0:
        torch.cuda.cudart().cudaProfilerStart()
        nvtx.range_push("train")
    # with deepspeed.zero.Init(remote_device="nvme", config_dict_or_path=ds_config):
    # model = LlamaForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
    model = AutoModelForCausalLM.from_pretrained(opt_model, ignore_mismatched_sizes=True).to(device).train()
    # set AC no-complie
    # model.gradient_checkpointing_enable() 
    
    # model.config.use_cache = False

    # # # set Activation checkpointing
    print("BEFORE START")

    ## ----------SAC-------------
    # model.config.use_cache = False   # SAC 가 KV-cache와 충돌하지 않도록
    # # ───────── SAC ① 전역 예산 켜기 ─────────
    import torch._functorch.config as torchfunc
    # torchfunc.activation_memory_budget = 0
    print("AC ENABLED")
    

    # deepspeed.runtime.zero.stage3.estimate_zero3_model_states_mem_needs_all_live(model, 1, 1)

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
    num_epochs = 1
    logger.info("Starting training loop...")
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1} start")
        # 프로파일링: local_rank==0 에서만
        for step, batch in enumerate(train_loader):
            st = time.time()
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels   = batch["labels"].to(device)

            
            outputs = model_engine(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()

            logger.info(f"Epoch {epoch+1} Step {step} Loss {loss.item():.4f}")
            print("EMPTY.")
            torch.cuda.synchronize()
            # get_accelerator().empty_cache()
            ed = time.time() - st
            training_time+=ed
            logger.info(f"[MAIN] Step Time: {ed:.4f}s")

            if max_step <= step+1:
                break

        logger.info(f"Epoch {epoch+1} completed.")
        
    logger.info(f"training_time: {training_time:.4f}s")
    if local_rank == 0:
        nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

    logger.info("Training finished (checkpoint skipped).")

if __name__ == "__main__":
    main()
