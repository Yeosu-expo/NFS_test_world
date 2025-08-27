import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def cosine_map_image(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, out_path: str = 'cosine_map.png') -> torch.Tensor:
    """Compute cosine similarity along a given dimension and save a grayscale image.
    Mapping: -1 → white, +1 → black.
    Returns the cosine similarity tensor.
    """
    # cosine similarity along the chosen vector dimension
    cos = F.cosine_similarity(x1, x2, dim=dim)

    # Map cosine values v∈[-1,1] to intensity I∈[0,1] with -1→1(white), +1→0(black)
    # I = (1 - v) / 2
    im = (1 - cos).div(2).clamp(0, 1)

    # Convert to uint8 grayscale image and save
    im_np = (im.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(im_np, mode='L')
    img.save(out_path)
    return cos


def cosine_flatten_vectors(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """두 입력 텐서를 1-D로 펼친 뒤 코사인 유사도를 계산하여 float으로 반환한다.
    - 입력은 어떤 shape든 허용 (이미 1-D여도 OK)
    - 서로 다른 dtype은 float32로 맞춤
    - 영노름 방지를 위해 분모에 작은 epsilon 적용
    """
    a = v1.reshape(-1).to(torch.float32)
    b = v2.reshape(-1).to(torch.float32)
    if a.numel() != b.numel():
        raise ValueError(f"Size mismatch: {a.numel()} vs {b.numel()}")
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    cos = torch.dot(a, b) / denom
    return float(cos.item())


# --- Inserted functions ---

def zscore_split_indices(x1: torch.Tensor, x2: torch.Tensor, tol: float = 0.5):
    """
    두 텐서를 전역 Z-Score 정규화한 뒤, 원소 페어별로 |z1 - z2| <= tol 이면 "비슷함"으로 분류합니다.
    반환: (similar_indices, dissimilar_indices)
    - similar_indices / dissimilar_indices: 다차원 인덱스 튜플의 리스트(list[tuple[int,...]]).
    - tol: 허용 오차(표준편차 단위). 기본 0.5.
    """
    if x1.shape != x2.shape:
        raise ValueError(f"Shape mismatch: {tuple(x1.shape)} vs {tuple(x2.shape)}")
    a = x1.float()
    b = x2.float()
    a = (a - a.mean()) / (a.std().clamp_min(1e-12))
    b = (b - b.mean()) / (b.std().clamp_min(1e-12))

    diff = (a - b).abs()
    similar_mask = torch.isfinite(diff) & (diff <= tol)
    dissimilar_mask = torch.isfinite(diff) & ~similar_mask

    sim_idx = [tuple(ix.tolist()) for ix in torch.nonzero(similar_mask, as_tuple=False)]
    dis_idx = [tuple(ix.tolist()) for ix in torch.nonzero(dissimilar_mask, as_tuple=False)]
    return sim_idx, dis_idx


def overlap_ratios(sim_a, dis_a, sim_b, dis_b):
    """
    네 개의 인덱스 리스트를 받아 겹침 비율을 계산합니다.
    - sim_a, sim_b: 서로 다른 비교에서 얻은 "비슷한" 인덱스 리스트
    - dis_a, dis_b: 서로 다른 비교에서 얻은 "비슷하지 않은" 인덱스 리스트
    반환: (ratio_similar, ratio_dissimilar)
      · ratio = |교집합| / |합집합|  (Jaccard 비율, 0~1)
    """
    set_sim_a = set(sim_a)
    set_sim_b = set(sim_b)
    set_dis_a = set(dis_a)
    set_dis_b = set(dis_b)

    inter_sim = len(set_sim_a & set_sim_b)
    union_sim = len(set_sim_a | set_sim_b)
    ratio_sim = (inter_sim / union_sim) if union_sim > 0 else 0.0

    inter_dis = len(set_dis_a & set_dis_b)
    union_dis = len(set_dis_a | set_dis_b)
    ratio_dis = (inter_dis / union_dis) if union_dis > 0 else 0.0
    return ratio_sim, ratio_dis


if __name__ == "__main__":
    # Example tensors
    x1 = torch.randn([512, 16, 512])
    x2 = torch.randn([512, 16, 512])
    # x1 = torch.ones_like(x1)
    # x2 = torch.ones_like(x2)

    # 1) Cosine similarity map (dim=1) → image
    cos_map = cosine_map_image(x1, x2, dim=1, out_path='cosine_map.png')
    print(cos_map)
    print(cos_map.shape)
    print('Saved cosine similarity image to cosine_map.png')

    # 2) Flattened cosine similarity (single scalar)
    cos_flat = cosine_flatten_vectors(x1, x2)
    print(f'Flattened cosine similarity: {cos_flat:.6f}')

    # 3) Pattern-wise split & overlap demo
    sim1, dis1 = zscore_split_indices(x1, x2, tol=0.5)
    # 두 번째 비교를 위해 약간의 잡음을 추가한 텐서쌍을 사용
    x1b = x1 + 0.1 * torch.randn_like(x1)
    x2b = x2 + 0.1 * torch.randn_like(x2)
    sim2, dis2 = zscore_split_indices(x1b, x2b, tol=0.5)

    r_sim, r_dis = overlap_ratios(sim1, dis1, sim2, dis2)
    print(f"similar overlap ratio (Jaccard): {r_sim:.4f}")
    print(f"dissimilar overlap ratio (Jaccard): {r_dis:.4f}")