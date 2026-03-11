from pathlib import Path

import numpy as np
import torch
from PIL import Image


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(image_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    image_np = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)


def save_reconstruction(image_tensor: torch.Tensor, path: Path):
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
    image_np = (
        image_tensor[0]
        .permute(1, 2, 0)
        .detach()
        .to("cpu", dtype=torch.float32)
        .numpy()
    )
    image_np = (image_np * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_np).save(path)
