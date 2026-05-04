from pathlib import Path
import json

import torch


def normalize_token_tensor(tokens: torch.Tensor) -> torch.Tensor:
    """Normalize transmitted token IDs into the shape expected by TiTok.decode_tokens()."""
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(1)
    if tokens.ndim != 3:
        raise ValueError(f"Expected token tensor with 1, 2, or 3 dims, got shape {tuple(tokens.shape)}")
    return tokens.to(torch.long)


def load_token_json(path: Path) -> torch.Tensor:
    payload = json.loads(path.read_text())
    return normalize_token_tensor(torch.tensor(payload["tokens"], dtype=torch.long))


def decode_token_ids(tokenizer, token_ids: torch.Tensor, device: str) -> torch.Tensor:
    token_ids = normalize_token_tensor(token_ids).to(device)
    return tokenizer.decode_tokens(token_ids)
