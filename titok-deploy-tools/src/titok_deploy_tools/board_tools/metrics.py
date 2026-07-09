from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    af = a.astype(np.float32).reshape(-1)
    bf = b.astype(np.float32).reshape(-1)
    d = bf - af
    ad = np.abs(d)
    return {
        "exact_equal": bool(np.array_equal(a, b)),
        "exact_match_count": int(np.sum(a.reshape(-1) == b.reshape(-1))),
        "numel": int(af.size),
        "max_abs_error": float(ad.max()) if ad.size else 0.0,
        "mean_abs_error": float(ad.mean()) if ad.size else 0.0,
        "rmse": float(math.sqrt(float(np.mean(d * d)))) if d.size else 0.0,
        "cosine_similarity": float(np.dot(af, bf) / max(float(np.linalg.norm(af) * np.linalg.norm(bf)), 1e-12))
        if af.size
        else 1.0,
    }


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def ssim_simple(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().to(torch.float64)
    y = b.detach().to(torch.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    values = []
    for channel in range(x.shape[1]):
        xc = x[:, channel].reshape(-1)
        yc = y[:, channel].reshape(-1)
        mux = xc.mean()
        muy = yc.mean()
        vx = ((xc - mux) ** 2).mean()
        vy = ((yc - muy) ** 2).mean()
        cov = ((xc - mux) * (yc - muy)).mean()
        values.append(float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2))))
    return float(sum(values) / len(values))


def token_pair_metrics(ref_tokens: torch.Tensor, ref_top5: torch.Tensor, cand_tokens: torch.Tensor, cand_top5: torch.Tensor) -> dict[str, float]:
    exact = ref_tokens == cand_tokens
    top5_match = (ref_top5 == cand_tokens.unsqueeze(-1)).any(dim=-1)
    return {
        "token_exact_fraction": float(exact.to(torch.float32).mean().item()),
        "token_mismatch_count": int((~exact).sum().item()),
        "token_top5_fraction": float(top5_match.to(torch.float32).mean().item()),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
