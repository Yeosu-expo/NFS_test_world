import os
import sys
import time
import math
import collections
from typing import List, Dict, Tuple

import matplotlib
matplotlib.use('Agg')  # headless plotting
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributed as dist

# Project utils (stay consistent with other scripts)
sys.path.insert(0, "/home/deepspeed")
from deep_utils.set_model import get_model, get_tokenizer, login_hf  # noqa: E402
from deep_utils.set_datasets import get_data_loader  # noqa: E402
from deep_utils.utils import get_args, get_logger, get_samples, push_range, pop_range  # noqa: E402


rank = None
local_rank = None
device = None
world_size = None


def ddp_init():
    global rank, local_rank, device, world_size
    rank_env = os.environ.get("RANK")
    local_rank_env = os.environ.get("LOCAL_RANK")
    world_size_env = os.environ.get("WORLD_SIZE")

    if rank_env is None or world_size_env is None:
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

    rank = int(rank_env)
    local_rank = int(local_rank_env) if local_rank_env is not None else 0
    world_size = int(world_size_env)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "eth0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_DEBUG", "INFO")


_LAYER_NAME_PAT = None
import re as _re
_LAYER_NAME_PAT = _re.compile(r"(?:layers|h|block|blocks|layer|encoder\.layer|decoder\.layer)\.(\d+)")


def _layer_id_from_param_name(name: str) -> int:
    m = _LAYER_NAME_PAT.search(name)
    if m:
        return int(m.group(1))
    for tok in name.split('.'):
        if tok.isdigit():
            return int(tok)
    return -1


def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = a.float(); b = b.float()
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1)
    den = torch.clamp(den, min=eps)
    return num / den


def _layer_hidden_vectors(module: nn.Module, hidden_size: int) -> collections.OrderedDict[int, torch.Tensor]:
    """Return ordered dict: layer_id -> 2D tensor [num_vectors, hidden_size].
    It slices gradients into vectors of length hidden_size along a suitable axis.
    Only parameters with floating grads are considered. Non-matching shapes are skipped.
    """
    buckets: Dict[int, List[torch.Tensor]] = collections.defaultdict(list)
    for name, p in module.named_parameters():
        g = p.grad
        if g is None:
            continue
        if not g.is_floating_point():
            continue
        gt = g.detach()
        if gt.numel() == 0:
            continue
        shape = list(gt.shape)
        if len(shape) == 1:
            if shape[0] == hidden_size:
                mat = gt.reshape(1, hidden_size).float()
            else:
                continue
        else:
            # Find an axis whose size is a multiple of hidden_size; prefer last
            axis = None
            if shape[-1] % hidden_size == 0:
                axis = len(shape) - 1
            else:
                for i, s in enumerate(shape):
                    if s % hidden_size == 0:
                        axis = i
                        break
            if axis is None:
                continue
            if axis != len(shape) - 1:
                # move selected axis to last
                perm = [i for i in range(len(shape)) if i != axis] + [axis]
                gt = gt.permute(*perm).contiguous()
                shape = list(gt.shape)
            last = shape[-1]
            if last % hidden_size != 0:
                continue
            chunks = last // hidden_size
            new_shape = (-1, chunks, hidden_size)
            try:
                gt2 = gt.reshape(new_shape)
            except Exception:
                gt2 = gt.contiguous().view(new_shape)
            mat = gt2.reshape(-1, hidden_size).float()
        lid = _layer_id_from_param_name(name)
        buckets[lid].append(mat)
    out: collections.OrderedDict[int, torch.Tensor] = collections.OrderedDict()
    for lid in sorted(buckets.keys()):
        out[lid] = torch.cat(buckets[lid], dim=0) if buckets[lid] else torch.empty(0, hidden_size, device=device, dtype=torch.float32)
    return out


def _gather_equal_shape(m: torch.Tensor) -> List[torch.Tensor]:
    """All-gather a tensor assuming identical shape across ranks."""
    gathered = [torch.empty_like(m) for _ in range(world_size)]
    dist.all_gather(gathered, m)
    return gathered


def _layer_sims_mean(module: nn.Module, hidden_size: int, logger) -> Dict[int, torch.Tensor]:
    """Compute per-vector average cosine similarity across ranks for each layer.
    Returns on rank 0: {layer_id: tensor[N] with mean cosine vs other ranks}; others return {}.
    """
    sims_by_layer: Dict[int, torch.Tensor] = {}
    mats = _layer_hidden_vectors(module, hidden_size)
    if world_size == 1:
        for lid, mat in mats.items():
            N = mat.shape[0]
            sims_by_layer[lid] = torch.ones(N, dtype=torch.float32, device=mat.device)
        return sims_by_layer

    # Validate row counts match across ranks per layer by gathering lengths then the matrix
    for lid, mat in mats.items():
        rows = torch.tensor([mat.shape[0]], device=device, dtype=torch.int64)
        rows_g = [torch.empty_like(rows) for _ in range(world_size)]
        dist.all_gather(rows_g, rows)
        row_counts = [int(t.item()) for t in rows_g]
        if any(rc != row_counts[0] for rc in row_counts):
            if rank == 0:
                logger.warning(f"[SIM] Skip layer {lid}: row-count mismatch across ranks {row_counts}")
            continue
        # Gather matrices
        mats_g = _gather_equal_shape(mat)
        if rank == 0:
            ref = mats_g[0]  # [N,H]
            N = ref.shape[0]
            if N == 0:
                sims_by_layer[lid] = torch.empty(0, dtype=torch.float32, device=ref.device)
                continue
            sims_sum = torch.zeros(N, dtype=torch.float32, device=ref.device)
            cnt = 0
            for r in range(1, world_size):
                s = _cosine(ref, mats_g[r])  # [N]
                sims_sum += s
                cnt += 1
            sims_mean = sims_sum / max(1, cnt)
            sims_by_layer[lid] = sims_mean
    return sims_by_layer


def save_summary_plot_multi(history_by_bound: Dict[float, Dict[int, List[float]]], outdir: str, title: str):
    if rank != 0:
        return
    _ensure_outdir(outdir)
    # Union of layers across bounds
    layer_set = set()
    for b in history_by_bound:
        layer_set.update(history_by_bound[b].keys())
    layers = sorted(layer_set)
    x = list(range(len(layers)))
    plt.figure()
    for b in history_by_bound:
        hist = history_by_bound[b]
        means, stds = [], []
        for lid in layers:
            vals = hist.get(lid, [])
            if not vals:
                means.append(0.0); stds.append(0.0)
            else:
                t = torch.tensor(vals, dtype=torch.float32)
                means.append(float((t.mean() * 100.0).item()))
                stds.append(float((t.std(unbiased=False) * 100.0).item()))
        plt.errorbar(x, means, yerr=stds, fmt='o-', capsize=3, label=f'bound={b:.2f}')
    plt.ylim(0.0, 100.0)
    plt.xticks(x, [str(l) for l in layers])
    plt.xlabel('layer index')
    plt.ylabel('percentage of similar vectors (%)')
    plt.title(title)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'grad_similar_percentage_summary.png'))
    plt.close()


def main():
    args = get_args()
    logger = get_logger(args.log_file)

    ddp_init()
    if rank == 0:
        logger.info(f"RANK: {rank}, WORLD_SIZE: {world_size}")

    push_range("grad_decomp_by_simil", rank, args.use_nsys)

    # Login/tokenizer/model
    login_hf(args.hug_token)
    tokenizer = get_tokenizer(args.model)

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

    pretrain_flag = (True if args.pretrain == 1 else False)
    model = get_model(args.model, tokenizer, device, dtype=torch.float32, pretrain=pretrain_flag)
    hidden_size = None
    try:
        hidden_size = int(getattr(getattr(model, 'config', object()), 'hidden_size', None) or 0)
    except Exception:
        hidden_size = 0
    if not hidden_size or hidden_size <= 0:
        try:
            emb = model.get_input_embeddings().weight
            hidden_size = int(emb.shape[1])
        except Exception:
            hidden_size = 0
    if hidden_size <= 0 and rank == 0:
        logger.warning("[SIM] Failed to determine hidden_size. Using 1024 as fallback.")
    if hidden_size <= 0:
        hidden_size = 1024

    # Wrap with DDP; we will use no_sync during backward to analyze pre-sync grads then manually allreduce
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

    bounds = getattr(args, 'similarity_bound', [0.9])
    if not isinstance(bounds, list):
        bounds = [float(bounds)]
    bounds = [max(0.0, min(1.0, float(b))) for b in bounds]
    # preserve order but unique
    seen = set(); ordered_bounds: List[float] = []
    for b in bounds:
        if b not in seen:
            seen.add(b)
            ordered_bounds.append(b)
    bounds = ordered_bounds
    if rank == 0:
        logger.info(f"[SIM] hidden_size={hidden_size}, bounds={bounds}")

    max_steps = int(args.max_steps)
    history_by_bound: Dict[float, Dict[int, List[float]]] = {b: {} for b in bounds}
    data_iter = iter(train_loader)
    model.train()

    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        input_ids, labels = get_samples(data_iter, train_loader, device)

        # Backward without DDP gradient sync to analyze pre-sync grads
        if world_size > 1:
            ctx = model.no_sync()
        else:
            # dummy context manager
            class _N:
                def __enter__(self):
                    return None
                def __exit__(self, exc_type, exc, tb):
                    return False
            ctx = _N()
        with ctx:
            with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=(device.type == 'cuda')):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
        # Unscale grads for analysis and manual sync
        scaler.unscale_(optimizer)

        # Analyze gradients on the underlying module
        module = model.module if hasattr(model, 'module') else model
        sims_by_layer = _layer_sims_mean(module, hidden_size, logger)
        if rank == 0:
            for lid, sims_mean in sims_by_layer.items():
                if sims_mean.numel() == 0:
                    for b in bounds:
                        history_by_bound[b].setdefault(lid, []).append(0.0)
                    continue
                for b in bounds:
                    frac = float((sims_mean >= b).float().mean().item())
                    history_by_bound[b].setdefault(lid, []).append(frac)

        # Manual allreduce to perform a real update (average across all ranks)
        if world_size > 1:
            for p in module.parameters():
                if p.grad is None:
                    continue
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)

        # Clip and optimizer step
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        try:
            if device.type == 'cuda':
                torch.cuda.synchronize()
        except Exception:
            pass
        dt = time.time() - t0
        if rank == 0:
            logger.info(f"[Step {step+1}/{max_steps}] loss={float(loss.item()):.6f} time={dt:.4f}s")

        # Occasional barrier to keep ranks aligned
        if world_size > 1 and ((step + 1) % 10 == 0):
            try:
                dist.barrier()
            except Exception:
                pass

    # Save summary plot (rank 0)
    outdir = os.path.join(os.getcwd(), 'alpha-code', 'testbed', 'output', 'grad_decomp_by_simil')
    save_summary_plot_multi(history_by_bound, outdir, title='Layer-wise similar-grad percentage')
    if rank == 0:
        logger.info(f"[PLOT] Saved summary to {os.path.join(outdir, 'grad_similar_percentage_summary.png')}")

    pop_range(rank, args.use_nsys)
    try:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    main()
