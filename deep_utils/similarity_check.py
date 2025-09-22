import torch
import deepspeed
from deepspeed import zero as ds_zero
import torch.distributed as dist
import logging
from deepspeed.runtime.zero.utils import is_zero_param

def _flatten_grads(engine: deepspeed.DeepSpeedEngine, logger:logging.Logger) -> torch.Tensor:
    """Concatenate all parameter grads into a 1-D float tensor on the same device.
    Works with DP/ZeRO; parameters with None grads are skipped.
    """
    device = engine.device

    def _collect(module) -> torch.Tensor:
        chunks = []
        for p in module.parameters():
            g = None
            # DeepSpeed/ZeRO: main_grad 우선
            if hasattr(p, 'main_grad') and p.main_grad is not None:
                g = p.main_grad
            elif p.grad is not None:
                g = p.grad
            elif hasattr(p, 'ds_tensor') and getattr(p, 'ds_tensor') is not None and getattr(p.ds_tensor, 'grad', None) is not None:
                # 일부 ZeRO 구현에서 shard 텐서의 grad가 여기에 존재
                g = p.ds_tensor.grad
            if g is None:
                continue
            g = g.detach()
            if not g.is_floating_point():
                g = g.float()
            chunks.append(g.reshape(-1))
        if len(chunks) == 0:
            return torch.empty(0, device=device, dtype=torch.float32)
        return torch.cat(chunks, dim=0)
    
    # 1) 일반 수집 시도
    flat = _collect(engine.module)
    if flat.numel() > 0:
        return flat

    # 2) ZeRO일 가능성: 파라미터를 모아(grads 포함) 다시 시도
    try:
        enabled = True
        # modifier_rank=None → 모든 rank에서 full param 보이도록 모음 (임시 메모리 비용 주의)
        params = list(engine.module.parameters())
        with ds_zero.GatheredParameters(params, modifier_rank=None, enabled=enabled):
            flat = _collect(engine.module)
        return flat
    except Exception as e:
        # 마지막 안전장치: 빈 텐서 반환
        if logger is not None and dist.is_initialized():
            if dist.get_rank() == 0:
                logger.warning(f"[GradCorr] GatheredParameters failed: {e}")
        return torch.empty(0, device=device, dtype=torch.float32)


def _pearson_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two 1-D tensors (same length).
    Implemented as cosine similarity of Z-scored vectors.
    Returns a Python float.
    """
    a = a.float(); b = b.float()
    a = a - a.mean()
    b = b - b.mean()
    a_std = a.std().clamp_min(1e-12)
    b_std = b.std().clamp_min(1e-12)
    a = a / a_std
    b = b / b_std
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float(torch.dot(a, b) / denom)


def measure_grad_corr(engine: deepspeed.DeepSpeedEngine, logger:logging.Logger):
    """
    Compute Pearson correlation between **rank 0** and every other rank using the
    full flattened gradient vector (no sampling). Logs results on rank 0 with tag [GradCorr].
    """
    if not dist.is_initialized() or dist.get_world_size() < 2:
        logger.warning("[GradCorr] Skipped: not enough size of dist group.")
        return

    gflat = _flatten_grads(engine, logger)
    if gflat.numel() == 0:
        logger.warning("[GradCorr] Skipped: no grad (ensure call AFTER backward() and BEFORE step(); ZeRO may shard grads).")
        return

    ws = dist.get_world_size()

    # Ensure all ranks have same length (typical DP). If lengths mismatch, return early.
    local_len = torch.tensor([gflat.numel()], device=gflat.device, dtype=torch.long)
    lens = [torch.zeros_like(local_len) for _ in range(ws)]
    dist.all_gather(lens, local_len)
    if any(int(l.item()) != int(local_len.item()) for l in lens):
        if dist.get_rank() == 0:
            logger.warning("[GradCorr] Skipped: gradient lengths differ across ranks (likely ZeRO sharding).")
        return

    # Gather full gradients from all ranks
    gathered = [torch.empty_like(gflat) for _ in range(ws)]
    dist.all_gather(gathered, gflat)

    # Rank 0 computes correlations
    if dist.get_rank() == 0:
        ref = gathered[0]
        N = ref.numel()
        for r in range(1, ws):
            corr = _pearson_1d(ref, gathered[r])
            logger.info(f"[GradCorr] N={N} corr(rank0, rank{r})={corr:.6f}")
