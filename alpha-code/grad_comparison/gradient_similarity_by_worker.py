import os
import sys
import time
import math
from typing import List
import re
import collections
import matplotlib
matplotlib.use('Agg')  # headless plotting
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.nn as nn

# project utils
sys.path.insert(0, "/home/deepspeed")
from deep_utils.set_model import get_model, get_tokenizer, login_hf  # noqa: E402
from deep_utils.set_datasets import get_data_loader  # noqa: E402
from deep_utils.utils import get_args, get_logger, get_samples, push_range, pop_range  # noqa: E402


rank = None
local_rank = None
device = None
world_size = None
# z-score control flag (set from args in main)
USE_ZSCORE = True


def ddp_init():
    global rank, local_rank, device, world_size
    # Torch/torchrun env vars are expected
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "eth0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_DEBUG", "INFO")


def _flatten_grads(module: nn.Module) -> torch.Tensor:
    """Concatenate all parameter grads into a 1-D float tensor on current device.
    Skips params with None grad.
    """
    chunks: List[torch.Tensor] = []
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not g.is_floating_point():
            g = g.float()
        chunks.append(g.reshape(-1))
    if len(chunks) == 0:
        return torch.empty(0, device=device, dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def _pearson_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Similarity between vectors.
    If USE_ZSCORE: Pearson correlation (cosine of Z-scored vectors).
    Else: plain cosine similarity on raw vectors.
    """
    a = a.float(); b = b.float()
    if USE_ZSCORE:
        a = a - a.mean()
        b = b - b.mean()
        a_std = a.std().clamp_min(1e-12)
        b_std = b.std().clamp_min(1e-12)
        a = a / a_std
        b = b / b_std
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float(torch.dot(a, b) / denom)


# ---------------- Layerwise helpers ----------------
_LAYER_NAME_PAT = re.compile(r"(?:layers|h|block|blocks|layer|encoder\.layer|decoder\.layer)\.(\d+)")

def _layer_id_from_param_name(name: str) -> int:
    """Heuristically extract a layer index from a parameter name.
    Returns -1 for params that don't belong to a numbered layer (e.g., embeddings, lm_head).
    """
    m = _LAYER_NAME_PAT.search(name)
    if m:
        return int(m.group(1))
    # fallback: first numeric token anywhere in name
    for tok in name.split('.'):
        if tok.isdigit():
            return int(tok)
    return -1


def _flatten_grads_by_layer(module: nn.Module) -> collections.OrderedDict[int, torch.Tensor]:
    """Return an ordered dict: layer_id -> 1-D grad vector (may be empty if no grads).
    Layer -1 aggregates non-layer params (embeddings, lm_head, etc.).
    """
    buckets: dict[int, list[torch.Tensor]] = collections.defaultdict(list)
    for name, p in module.named_parameters():
        g = p.grad
        if g is None:
            continue
        if not g.is_floating_point():
            g = g.float()
        buckets[_layer_id_from_param_name(name)].append(g.detach().reshape(-1))
    out: collections.OrderedDict[int, torch.Tensor] = collections.OrderedDict()
    for lid in sorted(buckets.keys()):
        out[lid] = torch.cat(buckets[lid], dim=0) if buckets[lid] else torch.empty(0, device=device, dtype=torch.float32)
    return out


def measure_grad_corr_layerwise(module: nn.Module, logger, step: int, history: dict[int, list[tuple[int, float]]]):
    """Compute layer-wise Pearson corr (rank0 vs others) **before** grad sync and append to history.
    Also validates that all ranks report the same number of layers and identical per-layer vector sizes.
    history: {layer_id: [(step, corr_avg_across_other_ranks), ...]}
    """
    if world_size < 2:
        return

    # 1) Local layerwise grads
    local = _flatten_grads_by_layer(module)
    layer_ids = list(local.keys())
    layer_count = torch.tensor([len(layer_ids)], device=device, dtype=torch.int64)

    # 2) Validate layer counts match across ranks
    counts = [torch.empty_like(layer_count) for _ in range(world_size)]
    dist.all_gather(counts, layer_count)
    counts_host = [int(t.item()) for t in counts]
    if any(c != counts_host[0] for c in counts_host):
        if rank == 0:
            logger.error(f"[GradCorr/LW] Layer-count mismatch across ranks: {counts_host}")
        return

    # 3) Validate per-layer vector lengths match across ranks
    lengths = torch.tensor([int(local[lid].numel()) for lid in layer_ids], device=device, dtype=torch.int64)
    lengths_gathered = [torch.empty_like(lengths) for _ in range(world_size)]
    dist.all_gather(lengths_gathered, lengths)
    equal_lengths = all(torch.equal(lengths_gathered[r], lengths_gathered[0]) for r in range(world_size))
    if not equal_lengths:
        if rank == 0:
            shape_rows = [t.tolist() for t in lengths_gathered]
            logger.error(f"[GradCorr/LW] Per-layer vector-length mismatch across ranks: {shape_rows}")
        return

    # 4) Compute corr per layer by gathering same-length vectors
    #    (we assume same ordering of layer_ids across ranks given identical models)
    for i, lid in enumerate(layer_ids):
        v = local[lid]
        if v.numel() == 0:
            continue
        # gather this layer's vector
        gathered = [torch.empty_like(v) for _ in range(world_size)]
        dist.all_gather(gathered, v)
        if rank == 0:
            ref = gathered[0]
            corrs = []
            for r in range(1, world_size):
                corrs.append(_pearson_1d(ref, gathered[r]))
            corr_avg = float(sum(corrs) / max(1, len(corrs)))
            history.setdefault(lid, []).append((step, corr_avg))
            logger.info(f"[GradCorr/LW] step={step} layer={lid} corr_avg={corr_avg:.6f}")


def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def save_layerwise_plots(history: dict[int, list[tuple[int, float]]], outdir: str, logger):
    """Create per-layer time-series plots (dotted) and a summary plot (mean±std with min/max band).
    Saves all figures under `outdir` (created if missing). Rank 0 only should call this.
    """
    if rank != 0:
        return
    _ensure_outdir(outdir)

    # Sort layers for consistent x-axis
    layers = sorted(history.keys())

    # 1) Per-layer time series (dotted)
    for lid in layers:
        if not history[lid]:
            continue
        steps = [s for (s, _) in history[lid]]
        vals = [v for (_, v) in history[lid]]
        plt.figure()
        plt.plot(steps, vals, linestyle=':', marker=None)
        plt.ylim(-1.0, 1.0)
        plt.xlabel('step')
        plt.ylabel(('pearson' if USE_ZSCORE else 'cosine') + ' (rank0 vs others)')
        plt.title(f'Layer {lid} correlation over steps')
        plt.grid(True, linestyle=':', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'layer_corr_timeseries_layer{lid}.png'))
        plt.close()

    # 2) Summary plot: x=layer index, show mean±std and min/max band across all steps
    if layers:
        means, stds, mins, maxs = [], [], [], []
        for lid in layers:
            vals = [v for (_, v) in history[lid]]
            if not vals:
                means.append(0.0); stds.append(0.0); mins.append(0.0); maxs.append(0.0)
            else:
                t = torch.tensor(vals, dtype=torch.float32)
                means.append(float(t.mean()))
                stds.append(float(t.std(unbiased=False)))
                mins.append(float(t.min()))
                maxs.append(float(t.max()))
        x = list(range(len(layers)))
        plt.figure()
        # min-max band
        plt.fill_between(x, mins, maxs, alpha=0.2, step='mid', label='min~max')
        # mean ± std error bars
        plt.errorbar(x, means, yerr=stds, fmt='o-', label='mean ± std')
        plt.ylim(-1.0, 1.0)
        plt.xticks(x, [str(l) for l in layers])
        plt.xlabel('layer index')
        plt.ylabel(('pearson' if USE_ZSCORE else 'cosine') + ' (rank0 vs others)')
        plt.title('Layer-wise correlation summary')
        plt.grid(True, linestyle=':', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'layer_corr_summary.png'))
        plt.close()


def measure_grad_corr_ddp(module: nn.Module, logger):
    """Gather full flattened grads from all ranks and log Pearson corr (rank0 vs others).
    This must be called AFTER backward() and BEFORE any gradient synchronization.
    In this script we use `no_sync()` to prevent DDP's implicit all-reduce, then
    perform a manual all-reduce after measuring.
    """
    if world_size < 2:
        return

    gflat = _flatten_grads(module)
    if gflat.numel() == 0:
        if rank == 0:
            logger.warning("[GradCorr] Skipped: no grad (call after backward(), before sync).")
        return

    # all_gather full vectors (same length across ranks in DP)
    gathered: List[torch.Tensor] = [torch.empty_like(gflat) for _ in range(world_size)]
    dist.all_gather(gathered, gflat)

    if rank == 0:
        ref = gathered[0]
        N = ref.numel()
        for r in range(1, world_size):
            corr = _pearson_1d(ref, gathered[r])
            logger.info(f"[GradCorr] N={N} corr(rank0, rank{r})={corr:.6f}")


def manual_allreduce_grads(module: nn.Module):
    """Average grads across DP ranks manually (since we used no_sync())."""
    for p in module.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def main():
    ddp_init()

    args = get_args()
    logger = get_logger(args.log_file)  # rank0: file+stdout, others: silent

    global USE_ZSCORE
    USE_ZSCORE = (True if args.z_score == 1 else False)
    if rank == 0:
        logger.info(f"[ARGS] z_score normalization: {USE_ZSCORE}")

    logger.info(f"RANK: {rank}, WORLD_SIZE: {world_size}")

    # HF login/tokenizer/model
    login_hf(args.hug_token)
    tokenizer = get_tokenizer(args.model)
    pretrain = (True if args.pretrain == 1 else False)
    logger.info(f"[MAIN] Pretrain Use: {pretrain} {args.pretrain}")
    model = get_model(
        model_type=args.model,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.float32,  # keep master params in FP32 for stability
        pretrain=pretrain,
    )
    # ---- Enable Gradient Checkpointing to reduce activation memory ----
    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            # Transformers: gradient checkpointing is incompatible with use_cache
            if hasattr(getattr(model, "config", object()), "use_cache"):
                try:
                    if model.config.use_cache:
                        model.config.use_cache = False
                        logger.info("[GC] Disabled config.use_cache for gradient checkpointing.")
                except Exception:
                    pass
            # Prefer non-reentrant where supported (more stable with AMP/FlashAttn in practice)
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                logger.info("[GC] Gradient checkpointing ENABLED (use_reentrant=False).")
            except TypeError:
                model.gradient_checkpointing_enable()
                logger.info("[GC] Gradient checkpointing ENABLED.")
        else:
            logger.warning("[GC] Model has no gradient_checkpointing_enable(); skipping.")
    except Exception as e:
        logger.warning(f"[GC] Failed to enable gradient checkpointing: {e}")

    # Wrap with DDP (disable broadcast_buffers to minimize extra sync)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    # Initialize layer-wise correlation history container (rank-agnostic)
    global _LW_HISTORY
    _LW_HISTORY = {}

    # Data
    batch_size = 10
    logger.info(f"[MAIN] Batch Size: {batch_size}")
    # Use project loader which already handles DistributedSampler when is_multi=True
    data_loader = get_data_loader(
        batch_size=batch_size,
        max_length=args.max_length,
        split_size=16000,
        tokenizer=tokenizer,
        world_size=world_size,
        rank=rank,
        is_multi=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=0.001)

    push_range("train", rank, args.use_nsys)

    max_step = args.max_steps
    logger.info(f"Target max_steps: {max_step}")
    training_time = 0.0
    step = 0
    data_iter = iter(data_loader)

    while step < max_step:
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        # Fetch batch
        input_ids, labels = get_samples(data_iter, data_loader, device)

        # AMP setup (BF16 if supported, else FP16 + GradScaler)
        use_bf16 = torch.cuda.is_bf16_supported()
        autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
        scaler = getattr(main, "_scaler", None)
        if scaler is None:
            scaler = torch.cuda.amp.GradScaler(enabled=not use_bf16)
            main._scaler = scaler

        # Prevent DDP implicit grad sync (we'll reduce manually)
        with ddp_model.no_sync():
            with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=True):
                outputs = ddp_model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()

        # Unscale before any manual grad ops (detect inf/NaN)
        scaler.unscale_(optimizer)

        # Optional: clip to tame spikes
        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)

        # Measure pre-reduction gradient similarity (use unscaled grads)
        measure_grad_corr_ddp(ddp_model, logger)
        # Layer-wise correlation & validation
        measure_grad_corr_layerwise(ddp_model, logger, step=step, history=_LW_HISTORY)

        # Manually average grads across ranks in FP32 to avoid overflow
        for p in ddp_model.parameters():
            if p.grad is None:
                continue
            g32 = p.grad.data.to(torch.float32)
            dist.all_reduce(g32, op=dist.ReduceOp.SUM)
            g32.div_(world_size)
            p.grad.data.copy_(g32.to(p.grad.dtype))

        # Optimizer step via scaler (skips if inf detected)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        torch.cuda.synchronize()
        dt = time.time() - t0
        training_time += dt
        if rank == 0:
            logger.info(f"Step {step}/{max_step} Loss {loss.item():.4f} | step_time {dt:.4f}s")

    pop_range(rank, args.use_nsys)

    if rank == 0:
        logger.info(f"Training finished. total_time={training_time:.4f}s")
    # Save plots to output/ (create if missing)
    save_layerwise_plots(_LW_HISTORY, outdir=os.path.join(os.getcwd(), 'output'), logger=logger)

    try:
        dist.barrier()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    main()
