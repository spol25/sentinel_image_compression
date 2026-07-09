from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_stats(tensor: torch.Tensor, *, first_n: int = 16) -> dict[str, Any]:
    value = tensor.detach().to("cpu")
    flat = value.reshape(-1)
    stats: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
    }
    if not value.numel():
        return stats

    if value.is_floating_point():
        numeric = flat.to(torch.float32)
        stats.update(
            {
                "min": float(numeric.min().item()),
                "max": float(numeric.max().item()),
                "mean": float(numeric.mean().item()),
                "std": float(numeric.std(unbiased=False).item()),
                f"first_{first_n}": [float(x) for x in numeric[:first_n].tolist()],
            }
        )
    else:
        numeric = flat.to(torch.int32)
        stats.update(
            {
                "min": int(numeric.min().item()),
                "max": int(numeric.max().item()),
                "mean": float(numeric.to(torch.float32).mean().item()),
                f"first_{first_n}": [int(x) for x in numeric[:first_n].tolist()],
            }
        )
    return stats


def save_float_npz(path: Path, tensor: torch.Tensor, *, key: str = "output") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: tensor.detach().to("cpu", dtype=torch.float32).contiguous().numpy()})


def save_int_npz(path: Path, tensor: torch.Tensor, *, key: str = "output_int") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().to("cpu").contiguous()
    if value.dtype not in (torch.int8, torch.uint8):
        value = value.to(torch.int16)
    np.savez(path, **{key: value.numpy()})


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
