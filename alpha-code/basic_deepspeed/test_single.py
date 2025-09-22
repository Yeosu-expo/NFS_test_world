import os
import time
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from transformers import AutoTokenizer, LlamaForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset
import datasets
from huggingface_hub import login
from torch.utils.data import DataLoader
import logging
import torch.cuda.nvtx as nvtx
import torch.distributed as dist

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
    
    # single-node 환경: local_rank 로 GPU 지정
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logger.info(f"Using device {device}")

    # 로그인, 토크나이저, collator
    hug_token = os.environ["HUG_TOKEN"]
    login(hug_token)
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 데이터 로드 & 전처리 (예제에선 64개만 사용)
    logger.info("Loading and tokenizing dataset...")
    train_dataset = load_wikitext(tokenizer, collator, max_length=512, split_size=64)
    
    # DeepSpeed 설정
    ds_config = "/home/deepspeed/config/ds_config_single.json"
    logger.info("Loading model under ZeRO init...")
    if local_rank == 0:
            torch.cuda.cudart().cudaProfilerStart()
            nvtx.range_push("train")
    # with deepspeed.zero.Init(remote_device="nvme", config_dict_or_path=ds_config):
    model = LlamaForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
    
    if deepspeed.checkpointing.is_configured():
        print("Checkpointing is configured.")
    else:
        print("Checkpointing is not configured.")    
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

    ## set AC
    # deepspeed.checkpointing.configure(
    #     mpu_=dist.get_global_rank(),
    #     partition_activations=True, 
    #     contiguous_checkpointing=True, 
    #     num_checkpoints=100, 
    #     checkpoint_in_cpu=True, 
    #     synchronize=True, 
    #     profile=False
    # )
    
    # 학습 루프
    num_epochs = 1
    logger.info("Starting training loop...")
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1} start")
        # 프로파일링: local_rank==0 에서만
        for step, batch in enumerate(train_loader):
            model_engine.zero_grad()
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels   = batch["labels"].to(device)

            outputs = model_engine(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()

            logger.info(f"Epoch {epoch+1} Step {step} Loss {loss.item():.4f}")
            print("EMPTY.")
            get_accelerator().empty_cache()

        if local_rank == 0:
            nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        logger.info(f"Epoch {epoch+1} completed")

    logger.info("Training finished (checkpoint skipped).")

if __name__ == "__main__":
    main()
