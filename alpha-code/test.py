import os
import time
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset
import datasets
from huggingface_hub import login
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
import csv
import torch.distributed as dist
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx

# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ["NCCL_DEBUG"]="INFO"
os.environ["DS_ACCELERATOR"] = "cuda"
# os.environ["NCCL_SOCKET_IFNAME"] = "eth0"



def load_wikitext(tokenizer, collator, max_length=None):

    def mask_tokens(x):
        input_ids, labels = collator.torch_mask_tokens(x['input_ids'], special_tokens_mask=x['special_tokens_mask'])
        return {
            "input_ids": input_ids,
            "labels": labels
        }

    wikitext = datasets.load_dataset("wikitext", "wikitext-2-v1")
    train_dataset = wikitext["train"]
    
    train_dataset = train_dataset.map(lambda x: tokenizer(x["text"], max_length=max_length, padding='max_length', truncation=True, return_tensors='pt', return_special_tokens_mask=True), batched=True)
    train_dataset.set_format(type="torch", columns=["input_ids", "special_tokens_mask"])
    if collator.mlm:
        train_dataset = train_dataset.map(mask_tokens, remove_columns=['special_tokens_mask'])
    else:
        train_dataset = train_dataset.map(lambda x: {
            "input_ids": x["input_ids"],
            "labels": x["input_ids"]
        })

    return train_dataset

# 로그 설정: "training.log" 파일에 INFO 레벨 이상의 로그를 기록 (기존 파일 덮어쓰기)
logging.basicConfig(
    filename="/root/training.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("deepspeed")
logger.setLevel(logging.DEBUG)


# 파일 핸들러를 생성하여 원하는 로그 파일에 기록하도록 설정 (예: "deepspeed.log")
file_handler = logging.FileHandler("/root/deepspeed_inter.log", mode="w")
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", 
                                   datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)

# 이미 다른 핸들러가 있다면 중복되지 않도록 확인 후 추가
if not logger.handlers:
    logger.info("THERE")
    logger.addHandler(file_handler)
else:
    # 필요시 기존 핸들러도 파일 핸들러로 교체할 수 있습니다.
    logger.info("HERE")
    logger.handlers = [file_handler]

# 이제 DeepSpeed 관련 로그는 "deepspeed.log" 파일에 기록됩니다.

# torch.distributed의 디버그 로그도 활성화
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
logger.info("Logger initialized.")

def main():
    # 초기화
    rank = os.environ["RANK"]
    world_size = os.environ["WORLD_SIZE"]
    master_addr = os.environ["MASTER_ADDR"]
    master_port  = os.environ["MASTER_PORT"]
    hug_token = os.environ["HUG_TOKEN"]
    logger.info(f"RANK: {rank}, WORLD_SIZE: {world_size}, MASTER_ADDR: {master_addr}, MASTER_PORT: {master_port}")
    dist.init_process_group("nccl")
    ds_config_path = "/home/deepspeed/config/ds_config.json"
    deepspeed.init_distributed(rank=int(rank), world_size=int(world_size))
    
    # 모델 및 토크나이저 로드
    opt_model = "facebook/opt-125m"
    model_name = "meta-llama/Llama-3.2-3B"
    login(hug_token)
    # tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    logger.info("Loading datasets...")   
    train_dataset = load_wikitext(tokenizer, collator, 512).select(range(64))

    # DeepSpeed 설정 파일 경로 (ds_config.json이 해당 경로에 있어야 합니다)
    logger.info("Loading model...")
    if int(rank) == 0:
        torch.cuda.cudart().cudaProfilerStart()
        nvtx.range_push("train")

    # with deepspeed.zero.Init(remote_device="cpu", config_dict_or_path=ds_config_path):
    with deepspeed.zero.Init():
        # model = AutoModelForCausalLM.from_pretrained(opt_model, ignore_mismatched_sizes=True)
        model = LlamaForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
    # model = LlamaForCausalLM.from_pretrained(model_name)

    logger.info("Initializing DeepSpeed engine...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config_path
    )
    logger.info("DeepSpeed engine initialized.")

    logger.info("Loading dataloader ...")
    # DataLoader 생성 (배치 크기는 필요에 따라 조정)
    num_epochs = 1
    batch_size = model_engine.train_micro_batch_size_per_gpu()
    logger.info(f"Batchs Size: {batch_size}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(train_dataset, num_replicas=int(world_size), rank=int(rank)),
        num_workers=int(world_size)
    )

    ## set AC
    deepspeed.checkpointing.configure(
        mpu_=rank,
        partition_activations=True, 
        contiguous_checkpointing=True, 
        num_checkpoints=100, 
        checkpoint_in_cpu=True, 
        synchronize=True, 
        profile=False
    )
    
    # 학습 루프
    from deepspeed.runtime.utils import memory_status
    logger.info("train start.")
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1} started.")

        for i, batch in enumerate(train_dataloader):
            # 배치 데이터를 DeepSpeed가 사용하는 GPU 장치로 이동
            device = torch.device("cuda", model_engine.local_rank)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model_engine(input_ids=input_ids, labels=labels)
            model_engine.backward(outputs.loss)
            model_engine.step()

            msg = f"Epoch {epoch+1}, i: {i}, micro step {model_engine.micro_steps}, global step {model_engine.global_steps}, Loss: {outputs.loss.item()}"
            logger.info(msg)

            if int(rank) == 0:
                memory_status("Memory stats after training step", dist.get_rank())
            deepspeed.get_accelerator().empty_cache()

        msg = f"Epoch {epoch+1} completed"
        logger.info(msg)

    if int(rank) == 0:
        nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

    # 체크포인트 저장 (스킵)
    checkpoint_dir = "./checkpoint"
    # model_engine.save_checkpoint(checkpoint_dir)
    logger.info(f"(Skipped) Training completed and checkpoint saved in {checkpoint_dir}.")

if __name__ == "__main__":
    main()

