import os
import sys
import time
import math
from typing import List, Tuple
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
# z-score flag kept for legacy plotting helpers (unused in this script)
USE_ZSCORE = True


def ddp_init():
    global rank, local_rank, device, world_size
    # Support both single-process and torchrun modes
    rank_env = os.environ.get("RANK")
    local_rank_env = os.environ.get("LOCAL_RANK")
    world_size_env = os.environ.get("WORLD_SIZE")

    if rank_env is None or world_size_env is None:
        # Single-process mode: no process group
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

    # torchrun/multi-process mode
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


def _flatten_grads(module: nn.Module) -> torch.Tensor:
    """(Legacy) Concatenate all parameter grads into a 1-D float tensor on current device.
    Skips params with None grad. Not used in new workflow.
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
    """(Legacy) Similarity between vectors. Not used in new workflow."""
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
    """(Legacy) Layer-wise corr. Kept for reference; not used now."""
    if world_size < 2:
        return
    local = _flatten_grads_by_layer(module)
    layer_ids = list(local.keys())
    layer_count = torch.tensor([len(layer_ids)], device=device, dtype=torch.int64)
    counts = [torch.empty_like(layer_count) for _ in range(world_size)]
    dist.all_gather(counts, layer_count)
    counts_host = [int(t.item()) for t in counts]
    if any(c != counts_host[0] for c in counts_host):
        if rank == 0:
            logger.error(f"[GradCorr/LW] Layer-count mismatch across ranks: {counts_host}")
        return
    lengths = torch.tensor([int(local[lid].numel()) for lid in layer_ids], device=device, dtype=torch.int64)
    lengths_gathered = [torch.empty_like(lengths) for _ in range(world_size)]
    dist.all_gather(lengths_gathered, lengths)
    equal_lengths = all(torch.equal(lengths_gathered[r], lengths_gathered[0]) for r in range(world_size))
    if not equal_lengths:
        if rank == 0:
            shape_rows = [t.tolist() for t in lengths_gathered]
            logger.error(f"[GradCorr/LW] Per-layer vector-length mismatch across ranks: {shape_rows}")
        return
    for lid in layer_ids:
        v = local[lid]
        if v.numel() == 0:
            continue
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
    """(Legacy) DDP full-grad corr; not used in new workflow."""
    if world_size < 2:
        return
    gflat = _flatten_grads(module)
    if gflat.numel() == 0:
        if rank == 0:
            logger.warning("[GradCorr] Skipped: no grad (call after backward(), before sync).")
        return
    gathered: List[torch.Tensor] = [torch.empty_like(gflat) for _ in range(world_size)]
    dist.all_gather(gathered, gflat)
    if rank == 0:
        ref = gathered[0]
        N = ref.numel()
        for r in range(1, world_size):
            corr = _pearson_1d(ref, gathered[r])
            logger.info(f"[GradCorr] N={N} corr(rank0, rank{r})={corr:.6f}")


def manual_allreduce_grads(module: nn.Module):
    """(Legacy) Manual all-reduce for grads; unused now."""
    for p in module.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def _train_one_run(
    model: nn.Module,
    tokenizer,
    device,
    logger,
    max_steps: int,
    data_loader=None,
    cache_batches: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
    use_cached: bool = False,
) -> tuple[list[float], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Train one run for max_steps and return (loss_history, cached_batches).
    If use_cached is False, reads from data_loader and appends CPU clones of (input_ids, labels)
    into cache_batches for reuse by the second run. If use_cached is True, ignores data_loader
    and replays from cache_batches.
    """
    model.train()

    # Enable gradient checkpointing to reduce activation memory if available
    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            if hasattr(getattr(model, "config", object()), "use_cache"):
                try:
                    if model.config.use_cache:
                        model.config.use_cache = False
                except Exception:
                    pass
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
    except Exception:
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    losses: list[float] = []
    cached: List[Tuple[torch.Tensor, torch.Tensor]] = cache_batches if cache_batches is not None else []

    use_bf16 = torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=not use_bf16)

    if not use_cached:
        assert data_loader is not None, "data_loader is required when not using cached batches"
        data_iter = iter(data_loader)

    step = 0
    while step < max_steps:
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        if use_cached:
            input_cpu, labels_cpu = cached[step]
            input_ids = input_cpu.to(device, non_blocking=True)
            labels = labels_cpu.to(device, non_blocking=True)
        else:
            input_ids, labels = get_samples(data_iter, data_loader, device)
            # store CPU copies for deterministic replay later
            cached.append((input_ids.detach().to('cpu'), labels.detach().to('cpu')))

        with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=True):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        dt = time.time() - t0
        losses.append(float(loss.item()))
        if rank == 0:
            logger.info(f"[Run] step {step}/{max_steps} loss={loss.item():.6f} time={dt:.4f}s")

    return losses, cached


def _save_loss_scatter(losses: list[float], outdir: str, filename: str, title: str, ylabel: str = 'loss'):
    _ensure_outdir(outdir)
    steps = list(range(1, len(losses) + 1))
    plt.figure()
    plt.scatter(steps, losses, s=12)
    plt.xlabel('step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename))
    plt.close()

def _save_two_series_plot(
    series1: list[float],
    series2: list[float],
    label1: str,
    label2: str,
    outdir: str,
    filename: str,
    title: str,
    ylabel: str,
):
    _ensure_outdir(outdir)
    x = list(range(1, max(len(series1), len(series2)) + 1))
    plt.figure()
    if series1:
        plt.plot(range(1, len(series1) + 1), series1, label=label1, marker='o', markersize=2, linewidth=1)
    if series2:
        plt.plot(range(1, len(series2) + 1), series2, label=label2, marker='o', markersize=2, linewidth=1)
    plt.xlabel('step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename))
    plt.close()

def _next_batch(data_iter, data_loader):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        batch = next(data_iter)
    return batch, data_iter

def _collect_batches_iter(data_iter, data_loader, device, count: int) -> tuple[list[Tuple[torch.Tensor, torch.Tensor]], any]:
    cached: list[Tuple[torch.Tensor, torch.Tensor]] = []
    while len(cached) < count:
        batch, data_iter = _next_batch(data_iter, data_loader)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        cached.append((input_ids.detach().to('cpu'), labels.detach().to('cpu')))
    return cached, data_iter

def _train_two_runs_on_cached_batches(
    model1: nn.Module,
    model2: nn.Module,
    cached_batches: list[Tuple[torch.Tensor, torch.Tensor]],
    device,
    logger,
):
    model1.train(); model2.train()
    _maybe_enable_gc(model1, logger)
    _maybe_enable_gc(model2, logger)

    opt1 = torch.optim.AdamW(model1.parameters(), lr=0.001)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=0.001)

    use_bf16 = torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler1 = torch.cuda.amp.GradScaler(enabled=not use_bf16)
    scaler2 = torch.cuda.amp.GradScaler(enabled=not use_bf16)

    losses1: list[float] = []
    losses2: list[float] = []
    grad_l1_diffs: list[float] = []
    grad_cos_sims: list[float] = []
    grad_l1_diffs: list[float] = []

    for step in range(len(cached_batches)):
        t0 = time.time()
        opt1.zero_grad(set_to_none=True)
        opt2.zero_grad(set_to_none=True)

        input_cpu, labels_cpu = cached_batches[step]
        input_ids = input_cpu.to(device, non_blocking=True)
        labels = labels_cpu.to(device, non_blocking=True)

        # Save RNG states to mirror stochasticity
        cpu_state = torch.get_rng_state()
        cuda_state = None
        if torch.cuda.is_available():
            try:
                cuda_state = torch.cuda.get_rng_state(device)
            except Exception:
                try:
                    cuda_state = torch.cuda.get_rng_state()
                except Exception:
                    cuda_state = None

        with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=True):
            out1 = model1(input_ids=input_ids, labels=labels)
            loss1 = out1.loss
        try:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                try:
                    torch.cuda.set_rng_state(cuda_state, device=device)
                except Exception:
                    torch.cuda.set_rng_state(cuda_state)
        except Exception:
            pass
        with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=True):
            out2 = model2(input_ids=input_ids, labels=labels)
            loss2 = out2.loss

        scaler1.scale(loss1).backward()
        scaler2.scale(loss2).backward()
        scaler1.unscale_(opt1)
        scaler2.unscale_(opt2)
        # L1 difference and cosine similarity on unscaled grads using 1-D flatten like worker script
        gflat1 = _flatten_grads(model1)
        gflat2 = _flatten_grads(model2)
        if gflat1.numel() == 0 or gflat2.numel() == 0:
            l1 = 0.0
            cos = 0.0
        else:
            # L1 over flattened vectors
            l1 = float((gflat1 - gflat2).abs().sum().item())
            # Cosine/Pearson per worker script (uses USE_ZSCORE flag inside)
            cos = _pearson_1d(gflat1, gflat2)
        torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
        scaler1.step(opt1)
        scaler2.step(opt2)
        scaler1.update()
        scaler2.update()

        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        dt = time.time() - t0
        losses1.append(float(loss1.item()))
        losses2.append(float(loss2.item()))
        grad_l1_diffs.append(float(l1))
        grad_cos_sims.append(float(cos))
        if rank == 0:
            logger.info(f"[CachedPair] step {step+1}/{len(cached_batches)} loss1={loss1.item():.6f} loss2={loss2.item():.6f} grad_l1={l1:.6e} grad_cos={cos:.6f} time={dt:.4f}s")
    return losses1, losses2, grad_l1_diffs, grad_cos_sims

def _maybe_enable_gc(model, logger=None):
    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            if hasattr(getattr(model, "config", object()), "use_cache"):
                try:
                    if model.config.use_cache:
                        model.config.use_cache = False
                except Exception:
                    pass
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                if logger and rank == 0:
                    logger.info("[GC] Enabled gradient checkpointing (use_reentrant=False)")
            except TypeError:
                model.gradient_checkpointing_enable()
                if logger and rank == 0:
                    logger.info("[GC] Enabled gradient checkpointing")
    except Exception as e:
        if logger and rank == 0:
            logger.warning(f"[GC] Failed to enable gradient checkpointing: {e}")

def _pearson_grads_streaming(model_a: nn.Module, model_b: nn.Module) -> float:
    s1 = 0.0
    s2 = 0.0
    ss1 = 0.0
    ss2 = 0.0
    sp = 0.0
    n = 0
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        ga = pa.grad
        gb = pb.grad
        if ga is None or gb is None:
            continue
        ga = ga.detach().float()
        gb = gb.detach().float()
        s1 += float(ga.sum().item())
        s2 += float(gb.sum().item())
        ss1 += float((ga * ga).sum().item())
        ss2 += float((gb * gb).sum().item())
        sp += float((ga * gb).sum().item())
        n += ga.numel()
    if n == 0:
        return 0.0
    mean1 = s1 / n
    mean2 = s2 / n
    var1 = ss1 / n - mean1 * mean1
    var2 = ss2 / n - mean2 * mean2
    denom = math.sqrt(max(var1, 1e-12) * max(var2, 1e-12))
    cov = sp / n - mean1 * mean2
    return float(cov / denom) if denom > 0 else 0.0

def _train_two_runs_lockstep(
    model1: nn.Module,
    model2: nn.Module,
    device,
    logger,
    max_steps: int,
    data_loader,
):
    model1.train(); model2.train()
    _maybe_enable_gc(model1, logger)
    _maybe_enable_gc(model2, logger)

    opt1 = torch.optim.AdamW(model1.parameters(), lr=0.001)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=0.001)

    use_bf16 = torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler1 = torch.cuda.amp.GradScaler(enabled=not use_bf16)
    scaler2 = torch.cuda.amp.GradScaler(enabled=not use_bf16)

    data_iter = iter(data_loader)
    losses1: list[float] = []
    losses2: list[float] = []
    corrs: list[float] = []

    for step in range(max_steps):
        t0 = time.time()
        opt1.zero_grad(set_to_none=True)
        opt2.zero_grad(set_to_none=True)

        # Fetch one batch and feed both models identically
        input_ids, labels = get_samples(data_iter, data_loader, device)

        # Save RNG states so model2 sees identical stochastic ops (e.g., dropout)
        cpu_state = torch.get_rng_state()
        cuda_state = None
        if torch.cuda.is_available():
            try:
                cuda_state = torch.cuda.get_rng_state(device)
            except Exception:
                try:
                    cuda_state = torch.cuda.get_rng_state()
                except Exception:
                    cuda_state = None

        with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=True):
            out1 = model1(input_ids=input_ids, labels=labels)
            loss1 = out1.loss
        # Restore RNG states before model2 forward/backward to mirror stochasticity
        try:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                try:
                    torch.cuda.set_rng_state(cuda_state, device=device)
                except Exception:
                    torch.cuda.set_rng_state(cuda_state)
        except Exception:
            pass

        with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=True):
            out2 = model2(input_ids=input_ids, labels=labels)
            loss2 = out2.loss

        scaler1.scale(loss1).backward()
        scaler2.scale(loss2).backward()

        # Unscale before measuring grads
        scaler1.unscale_(opt1)
        scaler2.unscale_(opt2)

        # L1 difference on unscaled grads (sum of abs elementwise diff)
        l1 = 0.0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            g1 = p1.grad
            g2 = p2.grad
            if g1 is None or g2 is None:
                continue
            g1f = g1.detach().float()
            g2f = g2.detach().float()
            # accumulate as float on CPU to avoid overflow on device
            l1 += float((g1f - g2f).abs().sum().item())

        # Optional: clip
        torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)

        scaler1.step(opt1)
        scaler2.step(opt2)
        scaler1.update()
        scaler2.update()

        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        dt = time.time() - t0

        losses1.append(float(loss1.item()))
        losses2.append(float(loss2.item()))
        grad_l1_diffs.append(float(l1))
        if rank == 0:
            logger.info(
                f"[CachedPair] step {step+1}/{len(cached_batches)} loss1={loss1.item():.6f} loss2={loss2.item():.6f} grad_l1={l1:.6e} time={dt:.4f}s"
            )

    return losses1, losses2, grad_l1_diffs


def main():
    ddp_init()

    args = get_args()
    logger = get_logger(args.log_file)  # rank0: file+stdout, others: silent

    # Encourage deterministic behavior for reproducibility
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except Exception:
        pass

    # Log-only (legacy flag kept for completeness)
    global USE_ZSCORE
    USE_ZSCORE = (True if args.z_score == 1 else False)
    if rank == 0:
        logger.info(f"RANK: {rank}, WORLD_SIZE: {world_size}")
        logger.info(f"[ARGS] z_score normalization flag (unused): {USE_ZSCORE}")

    # Only rank 0 performs the two-process comparison to avoid DDP sync complexity
    if rank != 0:
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        return

    # HF login/tokenizer
    login_hf(args.hug_token)
    tokenizer = get_tokenizer(args.model)

    # Data loader for caching batches (local-only to ensure identical batches)
    batch_size = args.batch_size
    logger.info(f"[DATA] Batch Size: {batch_size}")
    data_loader = get_data_loader(
        batch_size=batch_size,
        max_length=args.max_length,
        split_size=16000,
        tokenizer=tokenizer,
        world_size=1,
        rank=0,
        is_multi=False,
    )

    max_step = int(args.max_steps)
    logger.info(f"[RUN] Target max_steps: {max_step}")

    # Prepare two distinct cached batch sequences: pair1 and pair2
    data_iter = iter(data_loader)
    cached_pair1, data_iter = _collect_batches_iter(data_iter, data_loader, device, max_step)
    cached_pair2, data_iter = _collect_batches_iter(data_iter, data_loader, device, max_step)

    # Initialize and train Pair 1 (same-seed two models on cached_pair1)
    pretrain_flag = (True if args.pretrain == 1 else False)
    logger.info(f"[PAIR1 INIT] pretrain flag: {pretrain_flag}")
    if pretrain_flag:
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        p1_m1 = get_model(args.model, tokenizer, device, dtype=torch.float32, pretrain=True)
        torch.manual_seed(65); torch.cuda.manual_seed_all(65)
        p1_m2 = get_model(args.model, tokenizer, device, dtype=torch.float32, pretrain=True)
    else:
        p1_m1 = get_model(args.model, tokenizer, device, dtype=torch.float32, pretrain=False)
        p1_m2 = get_model(args.model, tokenizer, device, dtype=torch.float32, pretrain=False)

    push_range("pair1_train", rank, args.use_nsys)
    p1_loss1, p1_loss2, p1_l1, p1_cos = _train_two_runs_on_cached_batches(p1_m1, p1_m2, cached_pair1, device, logger)
    pop_range(rank, args.use_nsys)

    # Cleanup pair1 models
    try:
        del p1_m1, p1_m2
        torch.cuda.empty_cache()
    except Exception:
        pass

    # Initialize and train Pair 2 (same-seed two models on cached_pair2)
    logger.info(f"[PAIR2 INIT] pretrain flag: {pretrain_flag}")
    if pretrain_flag:
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        p2_m1 = get_model(args.model, tokenizer, device, dtype=torch.float32, pretrain=True)
        torch.manual_seed(65); torch.cuda.manual_seed_all(65)
        p2_m2 = get_model(args.model, tokenizer, device, dtype=torch.float32, pretrain=True)
    else:
        p2_m1 = get_model(args.model, tokenizer, device, dtype=torch.float32, pretrain=False)
        p2_m2 = get_model(args.model, tokenizer, device, dtype=torch.float32, pretrain=False)

    push_range("pair2_train", rank, args.use_nsys)
    p2_loss1, p2_loss2, p2_l1, p2_cos = _train_two_runs_on_cached_batches(p2_m1, p2_m2, cached_pair2, device, logger)
    pop_range(rank, args.use_nsys)

    # Cleanup pair2 models
    try:
        del p2_m1, p2_m2
        torch.cuda.empty_cache()
    except Exception:
        pass

    outdir = os.path.join(os.getcwd(), 'output')
    _save_loss_scatter(p1_loss1, outdir, 'pair1_proc1_loss.png', title='Pair1 Model1 Loss per Step')
    _save_loss_scatter(p1_loss2, outdir, 'pair1_proc2_loss.png', title='Pair1 Model2 Loss per Step')
    _save_loss_scatter(p2_loss1, outdir, 'pair2_proc1_loss.png', title='Pair2 Model1 Loss per Step')
    _save_loss_scatter(p2_loss2, outdir, 'pair2_proc2_loss.png', title='Pair2 Model2 Loss per Step')
    # Combined gradient similarity plot: pair1 vs pair2
    _save_two_series_plot(
        p1_l1, p2_l1,
        label1='Process 1 (pair1) grad L1',
        label2='Process 2 (pair2) grad L1',
        outdir=outdir,
        filename='grad_l1_pair_compare.png',
        title='Gradient L1 difference per step (pair1 vs pair2)',
        ylabel='sum |g1 - g2|',
    )
    # Combined gradient cosine similarity plot: pair1 vs pair2
    _save_two_series_plot(
        p1_cos, p2_cos,
        label1='Process 1 (pair1) grad cos',
        label2='Process 2 (pair2) grad cos',
        outdir=outdir,
        filename='grad_cos_pair_compare.png',
        title='Gradient cosine similarity per step (pair1 vs pair2)',
        ylabel='cosine',
    )
    logger.info(f"[PLOTS] Saved 4 loss scatters and combined grad L1/cos plots to {outdir}")

    # Clean up DDP
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
