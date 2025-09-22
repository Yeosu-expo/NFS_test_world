import os
import sys
import time
from typing import List

import matplotlib
matplotlib.use('Agg')  # headless plotting
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributed as dist

# Project utils (keep consistent with existing scripts)
sys.path.insert(0, "/home/deepspeed")
from deep_utils.set_model import get_model, get_tokenizer, login_hf  # noqa: E402
from deep_utils.set_datasets import get_data_loader  # noqa: E402
from deep_utils.utils import get_args, get_logger, get_samples, push_range, pop_range  # noqa: E402


# Globals populated by ddp_init()
rank = None
local_rank = None
device = None
world_size = None


def ddp_init():
    """Initialize distributed environment if launched via torchrun.
    Keeps single-process CPU/GPU fallback for convenience.
    """
    global rank, local_rank, device, world_size

    rank_env = os.environ.get("RANK")
    local_rank_env = os.environ.get("LOCAL_RANK")
    world_size_env = os.environ.get("WORLD_SIZE")

    if rank_env is None or world_size_env is None:
        # Single-process mode
        rank = 0
        local_rank = int(local_rank_env) if local_rank_env is not None else 0
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(local_rank)
            except Exception:
                local_rank = 0
                torch.cuda.set_device(0)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        world_size = 1
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        return

    # torchrun / multi-process mode
    rank = int(rank_env)
    local_rank = int(local_rank_env) if local_rank_env is not None else 0
    world_size = int(world_size_env)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Reasonable defaults that have worked in other scripts here
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "eth0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_DEBUG", "INFO")


def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_loss_plot(losses: List[float], outdir: str, filename: str, title: str = "Training Loss"):
    if rank != 0:
        return
    _ensure_outdir(outdir)
    steps = list(range(1, len(losses) + 1))
    plt.figure()
    plt.plot(steps, losses, marker='o', linestyle='-')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title(title)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename))
    plt.close()


def main():
    args = get_args()
    logger = get_logger(args.log_file)

    ddp_init()

    if rank == 0:
        logger.info(f"RANK: {rank}, WORLD_SIZE: {world_size}")

    # Optional NVTX profiling block around the whole run
    push_range("baseline1", rank, args.use_nsys)

    # Tokenizer / HF login
    login_hf(args.hug_token)
    tokenizer = get_tokenizer(args.model)

    # Data loader (DistributedSampler inside)
    batch_size = args.batch_size
    if rank == 0:
        logger.info(f"[DATA] Batch Size: {batch_size}")

    train_loader = get_data_loader(
        batch_size=batch_size,
        max_length=args.max_length,
        split_size=16000,
        tokenizer=tokenizer,
        world_size=world_size,
        rank=rank,
        is_multi=(world_size > 1),
    )

    # Model (optionally from scratch)
    pretrain_flag = (True if args.pretrain == 1 else False)
    if rank == 0:
        logger.info(f"[MODEL] type={args.model} pretrain={pretrain_flag}")

    model = get_model(args.model, tokenizer, device, dtype=torch.float32, pretrain=pretrain_flag)

    # Wrap in DDP when using multi-process
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == 'cuda' else None,
            output_device=local_rank if device.type == 'cuda' else None,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    use_bf16 = torch.cuda.is_bf16_supported() if device.type == 'cuda' else False
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and not use_bf16))

    # Train
    max_steps = int(args.max_steps)
    if rank == 0:
        logger.info(f"[TRAIN] max_steps={max_steps}")

    data_iter = iter(train_loader)
    loss_history: List[float] = []

    model.train()
    # Enable gradient checkpointing if available to reduce memory
    try:
        if hasattr(model, 'module'):
            mod = model.module
        else:
            mod = model
        if hasattr(mod, "gradient_checkpointing_enable"):
            if hasattr(getattr(mod, "config", object()), "use_cache"):
                try:
                    if mod.config.use_cache:
                        mod.config.use_cache = False
                except Exception:
                    pass
            try:
                mod.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                mod.gradient_checkpointing_enable()
    except Exception:
        pass

    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        input_ids, labels = get_samples(data_iter, train_loader, device)

        with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=(device.type == 'cuda')):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # For logging, take average loss across workers
        loss_avg = loss.detach()
        if world_size > 1:
            loss_avg = loss_avg.clone()
            dist.all_reduce(loss_avg, op=dist.ReduceOp.SUM)
            loss_avg = loss_avg / world_size

        # Sync for accurate timings (best-effort)
        try:
            if device.type == 'cuda':
                torch.cuda.synchronize()
        except Exception:
            pass

        dt = time.time() - t0
        if rank == 0:
            loss_scalar = float(loss_avg.item())
            loss_history.append(loss_scalar)
            logger.info(f"[Step {step+1}/{max_steps}] loss={loss_scalar:.6f} time={dt:.4f}s")

    # Save plot (rank 0 only)
    outdir = os.path.join(os.getcwd(), 'alpha-code', 'grad_comparison', 'output', 'baseline1')
    _save_loss_plot(loss_history, outdir, 'loss.png', title='Baseline1: Training Loss')
    if rank == 0:
        logger.info(f"[PLOT] Saved loss curve to {os.path.join(outdir, 'loss.png')}")

    # Clean up
    pop_range(rank, args.use_nsys)
    try:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    main()
