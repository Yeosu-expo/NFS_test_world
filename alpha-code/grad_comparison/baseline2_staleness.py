import os
import sys
import time
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')  # headless plotting
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

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


def _save_loss_plot(losses: List[float], outdir: str, filename: str, title: str = "Training Loss (staleness sim)"):
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


def _allreduce_grads(model: nn.Module, group, div_factor: int):
    for p in model.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group)
        if div_factor > 1:
            p.grad.div_(div_factor)


def _flatten_grads(model: nn.Module, device: torch.device) -> torch.Tensor:
    """Flatten all parameter gradients into a single 1-D tensor (float32)."""
    grads = []
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        # Ensure float32 for numerical stability
        if g.dtype != torch.float32:
            g = g.float()
        grads.append(g.reshape(-1))
    if len(grads) == 0:
        return torch.zeros(0, device=device, dtype=torch.float32)
    return torch.cat(grads)


def main():
    args = get_args()
    logger = get_logger(args.log_file)

    ddp_init()

    if rank == 0:
        logger.info(f"RANK: {rank}, WORLD_SIZE: {world_size}")

    # Optional NVTX profiling block around the whole run
    push_range("baseline2_stalness", rank, args.use_nsys)

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
    try:
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(0)
    except Exception:
        pass

    def run_case(slow_node_count: int, staleness_k: int) -> Tuple[str, List[float]]:
        # Clamp and normalize inputs
        if world_size <= 1:
            slow_node = 0
            k_sync = 1
        else:
            slow_node = max(0, min(int(slow_node_count), world_size - 1))
            k_sync = max(1, int(staleness_k))
            # If there are no slow nodes, enforce fully synchronous every step
            if slow_node == 0:
                k_sync = 1

        # Model (optionally from scratch). We do NOT wrap with DDP because we implement custom sync.
        pretrain_flag = (True if args.pretrain == 1 else False)
        model = get_model(args.model, tokenizer, device, dtype=torch.float32, pretrain=pretrain_flag)

        # Enable gradient checkpointing if available to reduce memory
        try:
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

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        use_bf16 = torch.cuda.is_bf16_supported() if device.type == 'cuda' else False
        autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and not use_bf16))

        # Choose the last N ranks as slow nodes, keep rank 0 in fast group for logging convenience
        if world_size > 1:
            slow_ranks = set(range(max(0, world_size - slow_node), world_size))
        else:
            slow_ranks = set()
        fast_ranks = [r for r in range(world_size) if r not in slow_ranks]

        # Create subgroups for fast/slow ranks if needed
        fast_pg = None
        slow_pg = None
        if world_size > 1 and 1 <= len(fast_ranks) < world_size:
            fast_pg = dist.new_group(ranks=fast_ranks)
        if world_size > 1 and 1 <= len(slow_ranks) < world_size:
            slow_pg = dist.new_group(ranks=sorted(list(slow_ranks)))

        if rank == 0:
            logger.info(f"[CASE] slow_node={slow_node}, staleness={k_sync}")
            logger.info(f"[CASE] fast_ranks={fast_ranks}, slow_ranks={sorted(list(slow_ranks))}")

        # Train
        max_steps = int(args.max_steps)
        data_iter = iter(train_loader)
        loss_history: List[float] = []
        model.train()

        # Track cosine similarity of avg gradients (fast vs slow) at global sync steps
        sim_steps: List[int] = []
        sim_values: List[float] = []

        for step in range(max_steps):
            step_idx = step + 1
            global_sync = (k_sync <= 1) or (step_idx % k_sync == 0)
            is_fast_rank = (rank in fast_ranks) if world_size > 1 else True

            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)

            input_ids, labels = get_samples(data_iter, train_loader, device)

            with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=(device.type == 'cuda')):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

            # Backward and unscale before any manual comms
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Gradient sync policy
            if world_size > 1:
                if global_sync:
                    # Before global sync, compute cosine similarity of avg gradients (fast vs slow)
                    if len(fast_ranks) > 0 and len(slow_ranks) > 0 and slow_pg is not None:
                        # Flatten local grads
                        grad_vec = _flatten_grads(model, device=device)

                        # Fast group average (compute only on fast ranks; rank 0 is in fast group by construction)
                        fast_avg = grad_vec.clone()
                        if is_fast_rank:
                            if fast_pg is not None and len(fast_ranks) > 1:
                                dist.all_reduce(fast_avg, op=dist.ReduceOp.SUM, group=fast_pg)
                                fast_avg.div_(len(fast_ranks))
                        # If single fast rank, clone is already the average

                        # Slow group average computed on slow ranks, then broadcast to all via WORLD
                        slow_avg = torch.empty_like(grad_vec)
                        slow_root = min(slow_ranks)
                        if rank in slow_ranks:
                            slow_avg = grad_vec.clone()
                            if len(slow_ranks) > 1:
                                dist.all_reduce(slow_avg, op=dist.ReduceOp.SUM, group=slow_pg)
                                slow_avg.div_(len(slow_ranks))
                            # Now broadcast slow_avg from slow_root to everyone so rank 0 can read it
                            dist.broadcast(slow_avg, src=slow_root, group=dist.group.WORLD)
                        else:
                            # Receive from slow_root
                            dist.broadcast(slow_avg, src=slow_root, group=dist.group.WORLD)

                        if rank == 0:
                            # Compute cosine similarity on rank 0
                            if slow_avg.numel() > 0 and fast_avg.numel() == slow_avg.numel():
                                cos = F.cosine_similarity(fast_avg, slow_avg, dim=0, eps=1e-8).item()
                                sim_steps.append(step_idx)
                                sim_values.append(cos)
                                logger.info(f"[SIM] step {step_idx} fast-vs-slow cosine={cos:.6f}")

                    # Everyone participates: average across all ranks
                    _allreduce_grads(model, group=dist.group.WORLD, div_factor=world_size)
                else:
                    # Only fast group syncs and updates; slow ranks compute grads but do not sync/step
                    if is_fast_rank and fast_pg is not None and len(fast_ranks) > 0:
                        _allreduce_grads(model, group=fast_pg, div_factor=len(fast_ranks))
            # else: single-process, nothing to reduce

            # Clip and step based on policy
            do_step = True
            if world_size > 1 and not global_sync and not is_fast_rank:
                do_step = False  # slow ranks skip updates between global syncs

            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            # Optionally sync for timings
            try:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            except Exception:
                pass

            dt = time.time() - t0
            if rank == 0:
                # Log local rank-0 loss (not globally averaged to avoid extra comms during non-global steps)
                loss_scalar = float(loss.detach().item())
                loss_history.append(loss_scalar)
                mode = "GLOBAL" if global_sync else "FAST-ONLY"
                logger.info(f"[CASE slow={slow_node} K={k_sync}] step {step_idx}/{max_steps} mode={mode} loss={loss_scalar:.6f} time={dt:.4f}s")

            # Hard barrier only on global sync steps to align all ranks
            if world_size > 1 and global_sync:
                try:
                    dist.barrier()
                except Exception:
                    pass

        # Save per-case plots on rank 0
        if rank == 0:
            outdir = os.path.join(os.getcwd(), 'alpha-code', 'grad_comparison', 'output', 'staleness_simulation')
            _ensure_outdir(outdir)
            steps = list(range(1, len(loss_history) + 1))
            plt.figure()
            plt.plot(steps, loss_history, marker='o', linestyle='-')
            plt.xlabel('step')
            plt.ylabel('loss')
            plt.title(f'sn={slow_node}, K={k_sync}')
            plt.grid(True, linestyle=':', linewidth=0.5)
            plt.tight_layout()
            per_file = os.path.join(outdir, f'baseline2_sn{slow_node}_st{k_sync}.png')
            plt.savefig(per_file)
            plt.close()
            logger.info(f"[PLOT] Saved per-case plot to {per_file}")

            # Cosine similarity plot at global sync points (only when both groups exist)
            if len(sim_steps) > 0 and len(sim_values) == len(sim_steps):
                plt.figure()
                plt.plot(sim_steps, sim_values, marker='o', linestyle='-')
                plt.xlabel('step')
                plt.ylabel('cosine similarity')
                plt.title(f'CosSim fast vs slow — sn={slow_node}, K={k_sync}')
                plt.ylim([-1.0, 1.0])
                plt.grid(True, linestyle=':', linewidth=0.5)
                plt.tight_layout()
                sim_file = os.path.join(outdir, f'baseline2_sn{slow_node}_st{k_sync}_cosine.png')
                plt.savefig(sim_file)
                plt.close()
                logger.info(f"[PLOT] Saved cosine similarity plot to {sim_file}")

        # Cleanup before next case
        try:
            del optimizer
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

        if world_size > 1:
            try:
                dist.barrier()
            except Exception:
                pass

        label = f"slow={slow_node},K={k_sync}"
        return label, loss_history

    # Build case list from CLI (paired, not cross product)
    slow_nodes: List[int] = getattr(args, 'slow_node', [0])
    stalenesses: List[int] = getattr(args, 'staleness', [1])
    if not isinstance(slow_nodes, list):
        slow_nodes = [int(slow_nodes)]
    if not isinstance(stalenesses, list):
        stalenesses = [int(stalenesses)]

    # Unique preserve order
    uniq_slow_nodes: List[int] = []
    for sn in slow_nodes:
        sni = int(sn)
        if sni not in uniq_slow_nodes:
            uniq_slow_nodes.append(sni)

    case_specs: List[Tuple[int, int]] = []
    # Always include synchronous case if 0 is present
    if 0 in uniq_slow_nodes:
        case_specs.append((0, 1))
        uniq_slow_nodes = [s for s in uniq_slow_nodes if s != 0]

    if len(stalenesses) == 0:
        stalenesses = [1]

    # Prepare K list aligned to remaining slow nodes
    if len(stalenesses) == 1:
        ks_list = [int(stalenesses[0])] * len(uniq_slow_nodes)
    else:
        if len(stalenesses) < len(uniq_slow_nodes):
            ks_list = [int(k) for k in stalenesses] + [int(stalenesses[-1])] * (len(uniq_slow_nodes) - len(stalenesses))
        else:
            ks_list = [int(k) for k in stalenesses[:len(uniq_slow_nodes)]]

    for s, k in zip(uniq_slow_nodes, ks_list):
        case_specs.append((int(s), int(k)))

    if rank == 0:
        logger.info(f"[RUN] Total cases: {len(case_specs)} -> {case_specs}")

    case_labels: List[str] = []
    case_losses: List[List[float]] = []
    for sn, ks in case_specs:
        lbl, losses = run_case(sn, ks)
        if rank == 0:
            case_labels.append(lbl)
            case_losses.append(losses)

    # Save combined plot (rank 0 only)
    if rank == 0:
        outdir = os.path.join(os.getcwd(), 'output', 'staleness_simulation')
        _ensure_outdir(outdir)
        plt.figure()
        for lbl, losses in zip(case_labels, case_losses):
            steps = list(range(1, len(losses) + 1))
            plt.plot(steps, losses, marker='', linestyle='-', label=lbl)
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title('Staleness Simulation: Loss by Case')
        plt.grid(True, linestyle=':', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        out_file = os.path.join(outdir, 'baseline2.png')
        plt.savefig(out_file)
        plt.close()
        logger.info(f"[PLOT] Saved combined loss plot to {out_file}")

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
