import os
import sys
import time
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')  # headless plotting
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# Project utils (consistent with other scripts)
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


def _flatten_grads(model: nn.Module, device: torch.device) -> torch.Tensor:
    """Flatten all parameter gradients into a single 1-D tensor (float32)."""
    grads = []
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        if g.dtype != torch.float32:
            g = g.float()
        grads.append(g.reshape(-1))
    if len(grads) == 0:
        return torch.zeros(0, device=device, dtype=torch.float32)
    return torch.cat(grads)


def _group_avg(vec: torch.Tensor, group, group_size: int) -> torch.Tensor:
    """Compute average of `vec` across `group` (no-op if size==1)."""
    out = vec.clone()
    if group is not None and group_size > 1:
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        out.div_(group_size) # all reduce가 이미 평균을 내지 않나? 그러면 또 나누는 건데, 이 줄이 필요한게 맞는지 생각해봐
    return out


def _gather_group_vectors(vec: torch.Tensor, group, group_size: int) -> List[torch.Tensor]:
    """All-gather identical-shaped 1D vectors across the given group.
    Returns list of tensors (length==group_size) on every rank in that group.
    """
    if group is None or group_size <= 0:
        return []
    gather_list = [torch.empty_like(vec) for _ in range(group_size)]
    dist.all_gather(gather_list, vec, group=group)
    return gather_list


def _avg_pairwise_cosine(vectors: List[torch.Tensor], eps: float = 1e-8) -> float:
    """Compute average pairwise cosine similarity among given 1-D tensors.
    Returns NaN if fewer than 2 vectors or zero-sized.
    """
    n = len(vectors)
    if n < 2:
        return float('nan')
    # Stack to [n, d]
    mat = torch.stack(vectors, dim=0)
    if mat.numel() == 0:
        return float('nan')
    # Normalize rows
    norms = torch.norm(mat, dim=1, keepdim=True).clamp_min(eps)
    mat_norm = mat / norms
    # Cosine matrix
    cos_mat = mat_norm @ mat_norm.t()
    # exclude diagonal
    off_diag_sum = cos_mat.sum() - torch.diag(cos_mat).sum()
    pairs = n * (n - 1)
    return float(off_diag_sum / pairs)


def _allreduce_grads(model: nn.Module, group, div_factor: int):
    for p in model.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group)
        if div_factor > 1:
            p.grad.div_(div_factor) # 여기도 마찬가지로 중복으로 나누는 것이 의심됨


def main():
    args = get_args()
    logger = get_logger(args.log_file)

    ddp_init()

    if rank == 0:
        logger.info(f"RANK: {rank}, WORLD_SIZE: {world_size}")

    push_range("grad_comp_by_sn_fn", rank, args.use_nsys)

    # HF / tokenizer
    login_hf(args.hug_token)
    tokenizer = get_tokenizer(args.model)

    # Data loader
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

    def run_case(slow_node_count: int, staleness_k: int) -> None:
        # Resolve inputs
        if world_size <= 1:
            slow_node = 0
            k_sync = int(staleness_k)
        else:
            slow_node = max(0, min(int(slow_node_count), world_size - 1))
            k_sync = max(1, int(staleness_k))
            if slow_node == 0:
                k_sync = max(1, k_sync)

        pretrain_flag = (True if args.pretrain == 1 else False)
        model = get_model(args.model, tokenizer, device, dtype=torch.float32, pretrain=pretrain_flag)

        # Enable gradient checkpointing if available
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

        # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        use_bf16 = torch.cuda.is_bf16_supported() if device.type == 'cuda' else False
        autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and not use_bf16))

        # Partition: choose last N ranks as slow; keep rank 0 in fast group
        if world_size > 1:
            slow_ranks = set(range(max(0, world_size - slow_node), world_size))
        else:
            slow_ranks = set()
        fast_ranks = [r for r in range(world_size) if r not in slow_ranks]

        fast_pg = None
        slow_pg = None
        if world_size > 1 and 1 <= len(fast_ranks) < world_size:
            fast_pg = dist.new_group(ranks=fast_ranks)
        if world_size > 1 and 1 <= len(slow_ranks) < world_size:
            slow_pg = dist.new_group(ranks=sorted(list(slow_ranks)))

        if rank == 0:
            logger.info(f"[CASE] slow_node={slow_node}, staleness={k_sync}")
            logger.info(f"[CASE] fast_ranks={fast_ranks}, slow_ranks={sorted(list(slow_ranks))}")

        max_steps = int(args.max_steps)
        # Allow 0: fast-only updates from the start; slow nodes still compute at step 1 for metrics
        slow_sync_steps = max(0, int(getattr(args, 'slow_sync', 0)))
        # Slow nodes compute gradients up to this step (ensures at least 1 grad for metrics)
        last_slow_grad_step = max(1, slow_sync_steps)
        data_iter = iter(train_loader)
        model.train()

        # Metrics to plot by step
        steps_axis: List[int] = []
        cos_fast_slow: List[float] = []  # fast avg vs STALE slow avg
        cos_intra_fast: List[float] = []  # pairwise within fast
        cos_intra_slow: List[float] = []  # single point at freeze step; NaN otherwise
        # cached slow group state at freeze
        stale_slow_avg_vec = None
        slow_once_similarity = float('nan')

        for step in range(max_steps):
            step_idx = step + 1
            is_fast_rank = (rank in fast_ranks) if world_size > 1 else True
            is_slow_rank = (rank in slow_ranks) if world_size > 1 else False
            # Evaluation-only step: at multiples of K, do comparisons only (no sync/step)
            is_eval_only = (k_sync >= 1 and (step_idx % k_sync == 0))

            t0 = time.time()

            # Decide whether this rank computes gradients this step
            do_compute = True
            if world_size > 1 and is_slow_rank and step_idx > last_slow_grad_step:
                do_compute = False

            grad_vec = None
            if do_compute:
                # Zero only on ranks that will compute this step; preserve slow grads otherwise
                optimizer.zero_grad(set_to_none=True)
                input_ids, labels = get_samples(data_iter, train_loader, device)
                with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=(device.type == 'cuda')):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Compute flattened gradient for this rank
                grad_vec = _flatten_grads(model, device=device)
            else:
                # Safe placeholder for ranks that are not computing this step
                loss = torch.tensor(0.0, device=device)

            # --- Similarity measurements ---
            if world_size > 1 and len(fast_ranks) > 0 and len(slow_ranks) > 0 and fast_pg is not None and slow_pg is not None:
                slow_root = min(slow_ranks)

                # Intra-fast every step
                intra_fast_val = float('nan')
                if is_fast_rank and grad_vec is not None:
                    gathered_fast = _gather_group_vectors(grad_vec, group=fast_pg, group_size=len(fast_ranks))
                    if rank == 0:
                        intra_fast_val = _avg_pairwise_cosine(gathered_fast)

                # Capture slow state exactly at the last_slow_grad_step
                if step_idx == last_slow_grad_step:
                    if is_slow_rank and grad_vec is not None:
                        tmp = _group_avg(grad_vec.clone(), group=slow_pg, group_size=len(slow_ranks))
                        slow_avg = tmp
                        # share slow_avg and intra_slow scalar to world for caching on rank 0
                        dist.broadcast(slow_avg, src=slow_root, group=dist.group.WORLD)
                        gathered_slow = _gather_group_vectors(grad_vec, group=slow_pg, group_size=len(slow_ranks))
                        intra_slow_val = float('nan')
                        if rank == slow_root:
                            intra_slow_val = _avg_pairwise_cosine(gathered_slow)
                        intra_slow_tensor = torch.tensor([intra_slow_val if (rank == slow_root) else 0.0], device=device, dtype=torch.float32)
                        dist.broadcast(intra_slow_tensor, src=slow_root, group=dist.group.WORLD)
                    else:
                        # Receive from slow_root on fast ranks
                        slow_avg = torch.empty_like(_flatten_grads(model, device=device))
                        dist.broadcast(slow_avg, src=slow_root, group=dist.group.WORLD)
                        intra_slow_tensor = torch.empty(1, device=device, dtype=torch.float32)
                        dist.broadcast(intra_slow_tensor, src=slow_root, group=dist.group.WORLD)

                    if rank == 0:
                        stale_slow_avg_vec = slow_avg.detach().clone()
                        slow_once_similarity = float(intra_slow_tensor.item())

                # Cross-group cosine each step vs STALE slow avg
                cos_fs = float('nan')
                if is_fast_rank and grad_vec is not None:
                    fast_avg = _group_avg(grad_vec.clone(), group=fast_pg, group_size=len(fast_ranks))
                    if rank == 0 and stale_slow_avg_vec is not None and fast_avg.numel() == stale_slow_avg_vec.numel():
                        cos_fs = F.cosine_similarity(fast_avg, stale_slow_avg_vec, dim=0, eps=1e-8).item()

                if rank == 0:
                    steps_axis.append(step_idx)
                    cos_intra_fast.append(intra_fast_val)
                    # Only set slow intra at freeze step
                    if step_idx == last_slow_grad_step:
                        cos_intra_slow.append(slow_once_similarity)
                    else:
                        cos_intra_slow.append(float('nan'))
                    # Cross similarity becomes valid from freeze step onward
                    if step_idx >= last_slow_grad_step:
                        cos_fast_slow.append(cos_fs)
                    else:
                        cos_fast_slow.append(float('nan'))
                    logger.info(
                        f"[SIM] step {step_idx} cos(fast,STALE slow)={cos_fs if step_idx>=last_slow_grad_step else float('nan'):.6f} intra_fast={intra_fast_val:.6f} intra_slow_once={slow_once_similarity if step_idx==last_slow_grad_step else float('nan'):.6f}"
                    )

            # --- Gradient update policy ---
            # Steps <= slow_sync_steps: all ranks sync (WORLD) and step.
            # Steps > slow_sync_steps: only fast ranks sync among themselves and step. Slow ranks do nothing.
            # Eval-only steps skip all sync/step.
            sync_code_local = 0  # 0=NONE, 1=WORLD, 2=FAST_PG
            if world_size > 1 and not is_eval_only:
                if step_idx <= slow_sync_steps:
                    _allreduce_grads(model, group=dist.group.WORLD, div_factor=world_size)
                    sync_code_local = 1
                else:
                    if is_fast_rank and fast_pg is not None and len(fast_ranks) > 0:
                        _allreduce_grads(model, group=fast_pg, div_factor=len(fast_ranks))
                        sync_code_local = 2

            # Step decision
            do_step = True
            if is_eval_only:
                do_step = False
            elif world_size > 1 and (rank in slow_ranks) and (step_idx > slow_sync_steps):
                do_step = False  # slow ranks skip updates entirely after freeze

            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            # --- Per-step diagnostic logging (ranks participation) ---
            if world_size > 1:
                try:
                    did_fwd = 1 if do_compute else 0
                    did_bwd = 1 if do_compute else 0
                    # Verify whether gradients are present (to ensure preservation on frozen slow ranks)
                    has_grad_flag = 1 if any(p.grad is not None for p in model.parameters()) else 0
                    flags = torch.tensor([did_fwd, did_bwd, sync_code_local, has_grad_flag], device=device, dtype=torch.int32)
                    gathered = [torch.empty_like(flags) for _ in range(world_size)]
                    dist.all_gather(gathered, flags, group=dist.group.WORLD)
                    if rank == 0:
                        arr = torch.stack(gathered, dim=0).cpu().tolist()
                        fwd_ranks = [i for i, v in enumerate(arr) if v[0] == 1]
                        bwd_ranks = [i for i, v in enumerate(arr) if v[1] == 1]
                        sync_world = [i for i, v in enumerate(arr) if v[2] == 1]
                        sync_fast = [i for i, v in enumerate(arr) if v[2] == 2]
                        has_grad = [i for i, v in enumerate(arr) if v[3] == 1]
                        logger.info(
                            f"[STEP {step_idx:03d}] fwd={fwd_ranks} bwd={bwd_ranks} sync_world={sync_world} sync_fast={sync_fast} grad_present={has_grad}"
                        )
                except Exception:
                    pass

            try:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            except Exception:
                pass

            # Lightweight progress log
            if rank == 0:
                dt = time.time() - t0
                mode = "ALL-SYNC" if step_idx <= slow_sync_steps else "FAST-ONLY"
                logger.info(f"[CASE slow={slow_node} K={k_sync} SS={slow_sync_steps}] step {step_idx}/{max_steps} mode={mode} loss={float(loss.detach().item()):.6f} time={dt:.4f}s")

            # Optional barrier to align timings on eval-only boundary
            if world_size > 1 and is_eval_only:
                try:
                    if step_idx <= slow_sync_steps:
                        dist.barrier()
                    elif fast_pg is not None:
                        dist.barrier(group=fast_pg)
                except Exception:
                    pass

        # Plot results per case (rank 0)
        if rank == 0:
            outdir = os.path.join(os.getcwd(), 'home', 'deepspeed', 'alpha-code', 'grad_comparison', 'output', 'by_sn_fn')
            _ensure_outdir(outdir)

            # 1) fast-avg vs slow-avg cosine
            if steps_axis:
                plt.figure()
                plt.plot(steps_axis, cos_fast_slow, marker='o', linestyle='-')
                plt.xlabel('step')
                plt.ylabel('cosine similarity')
                plt.title(f'Fast-vs-Slow avg grad — sn={slow_node}, K={k_sync}')
                plt.ylim([-1.0, 1.0])
                plt.grid(True, linestyle=':', linewidth=0.5)
                plt.tight_layout()
                f1 = os.path.join(outdir, f'by_sn_fn_sn{slow_node}_st{k_sync}_fast_vs_slow.png')
                plt.savefig(f1)
                plt.close()
                logger.info(f"[PLOT] Saved {f1}")

            # 2) intra-fast average pairwise cosine
            if steps_axis:
                plt.figure()
                plt.plot(steps_axis, cos_intra_fast, marker='o', linestyle='-')
                plt.xlabel('step')
                plt.ylabel('avg pairwise cosine (fast group)')
                plt.title(f'Intra-fast similarity — sn={slow_node}, K={k_sync}, SS={slow_sync_steps}')
                # overlay slow-once similarity at freeze step if available
                if not (slow_once_similarity != slow_once_similarity):  # check not-NaN
                    plt.scatter([max(1, slow_sync_steps)], [slow_once_similarity], color='red', marker='x', label='slow intra (once)')
                    plt.legend()
                plt.ylim([-1.0, 1.0])
                plt.grid(True, linestyle=':', linewidth=0.5)
                plt.tight_layout()
                f2 = os.path.join(outdir, f'by_sn_fn_sn{slow_node}_st{k_sync}_intra_fast.png')
                plt.savefig(f2)
                plt.close()
                logger.info(f"[PLOT] Saved {f2}")

            # 3) intra-slow average pairwise cosine (single point at freeze)
            if steps_axis:
                plt.figure()
                plt.plot(steps_axis, cos_intra_slow, marker='o', linestyle='None')
                plt.xlabel('step')
                plt.ylabel('avg pairwise cosine (slow group)')
                plt.title(f'Intra-slow similarity — sn={slow_node}, K={k_sync}, SS={slow_sync_steps}')
                plt.ylim([-1.0, 1.0])
                plt.grid(True, linestyle=':', linewidth=0.5)
                plt.tight_layout()
                f3 = os.path.join(outdir, f'by_sn_fn_sn{slow_node}_st{k_sync}_intra_slow.png')
                plt.savefig(f3)
                plt.close()
                logger.info(f"[PLOT] Saved {f3}")

        # Cleanup per case
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

    # Build case list from CLI (paired, not cross product)
    slow_nodes: List[int] = getattr(args, 'slow_node', [0])
    stalenesses: List[int] = getattr(args, 'staleness', [1])
    if not isinstance(slow_nodes, list):
        slow_nodes = [int(slow_nodes)]
    if not isinstance(stalenesses, list):
        stalenesses = [int(stalenesses)]

    uniq_slow_nodes: List[int] = []
    for sn in slow_nodes:
        sni = int(sn)
        if sni not in uniq_slow_nodes:
            uniq_slow_nodes.append(sni)

    if len(stalenesses) == 0:
        stalenesses = [1]

    if len(stalenesses) == 1:
        ks_list = [int(stalenesses[0])] * len(uniq_slow_nodes)
    else:
        if len(stalenesses) < len(uniq_slow_nodes):
            ks_list = [int(k) for k in stalenesses] + [int(stalenesses[-1])] * (len(uniq_slow_nodes) - len(stalenesses))
        else:
            ks_list = [int(k) for k in stalenesses[:len(uniq_slow_nodes)]]

    if rank == 0:
        pairs = list(zip(uniq_slow_nodes, ks_list))
        get_logger(args.log_file).info(f"[RUN] Total cases: {len(pairs)} -> {pairs}")

    for sn, ks in zip(uniq_slow_nodes, ks_list):
        run_case(sn, ks)

    pop_range(rank, args.use_nsys)
    try:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    main()
