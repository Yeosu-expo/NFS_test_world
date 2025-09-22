import argparse
import os
import sys
import logging
import torch.cuda.nvtx as nvtx
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_nsys", "-n", type=int, default=0,
                        help="if set to 1, it will be profiled with nsys including nvtx (default=0)")
    parser.add_argument("--log_file", "-f", type=str, default="training.log",
                        help="로그 파일명 (default=train.log)")
    parser.add_argument("--model", "-m", type=str, default="llama",
                        help="model name. (default=Llama)(option: llama, opt)")
    parser.add_argument("--max_steps", "-s", type=int, default=1,
                        help="최대 배치 수 (default=1)")
    parser.add_argument("--local_steps", "-ls", type=int, default=1)
    parser.add_argument("--max_length", "-l", type=int, default=512,
                        help="sample length (default=1024)")
    parser.add_argument("--checkpoint", "-c", type=int, default=0,
                        help="if set to 0, will not checkpoint.(default=0)")
    parser.add_argument("--local_rank", "-rank", type=int, default=0,
                        help="local rank (default=0)")
    parser.add_argument("--hug_token", "-ht", type=str, default="")
    parser.add_argument("--pretrain", "-p", type=int, default=0)
    parser.add_argument("--z_score", "-z", type=int, default=1)
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--slow_node", "-sn", type=int, nargs='+', default=[0],
                        help="Number of slow workers. Accepts multiple values, e.g., -sn 0 1 2")
    parser.add_argument("--staleness", "-st", type=int, nargs='+', default=[1],
                        help="Global sync interval K. Accepts multiple values, e.g., -st 1 5 10")
    parser.add_argument("--similarity_bound", "-sb", type=float, nargs='+', default=[0.9],
                        help="Cosine similarity bound(s) (0.0~1.0). Accepts multiple values, e.g., -sb 0.8 0.9 0.95")
    parser.add_argument("--slow_sync", "-ss", type=int, default=0,
                        help="Number of initial steps where all nodes synchronize and update together. After this, slow nodes stop participating. Use 0 for fast-only updates from the start (slow nodes still compute 1st grad for metrics).")
    args = parser.parse_args()

    return args

def get_logger(file_name:str) -> logging.Logger:
    rank_env = os.environ.get("RANK", "0")
    try:
        rank_int = int(rank_env)
    except Exception:
        rank_int = 0
    is_main = (rank_int == 0)

    log_path = "/home/deepspeed/alpha-code/" + file_name
    
    if is_main:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(log_path, mode="w"),
                logging.StreamHandler(sys.stdout),  # rank0 콘솔에도 출력
            ],
            force=True,
        )
    else:
        # 다른 rank는 어떤 핸들러에도 쓰지 않도록 루트 로거를 NullHandler로 구성
        logging.basicConfig(
            level=logging.CRITICAL,
            handlers=[logging.NullHandler()],
            force=True,
        )

    logger = logging.getLogger("deepspeed")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []      # 전용 핸들러 제거
    logger.propagate = True   # 루트로 전파 (비-main rank는 NullHandler라 실질적으로 기록되지 않음)
    if is_main:
        logger.info("Logger initialized (rank 0 only writes).")
    return logger

def checkpoint(model_engine, do_checkpoint:int, step:int, logger:logging.Logger, dir="./checkpoint"):
    client_sd = {}
    if do_checkpoint == 1:
        client_sd['step'] =int(step+1)
        chpt_id = int(step+1)
        model_engine.save_checkpoint(save_dir = "./checkpoint", tag = chpt_id, client_state = client_sd, save_latest = True)

        logger.info(f"[MAIN] Successfully checkpointed to {dir}")

def get_samples(data_iter, data_loader, device):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        batch = next(data_iter)
    input_ids = batch["input_ids"].to(device)
    labels   = batch["labels"].to(device)

    return input_ids, labels

def push_range(alias:str, rank:int, use_nsys:bool):
    if use_nsys != 1 or rank != 0:
        return
    
    torch.cuda.cudart().cudaProfilerStart()
    nvtx.range_push(alias)

def pop_range(rank:int, use_nsys:bool):
    if use_nsys != 1 or rank != 0:
        return
    
    nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
