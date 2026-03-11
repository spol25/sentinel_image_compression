import torch
import torch.nn as nn


class TiTokTokenEncoder(nn.Module):
    """Minimal TiTok inference wrapper that returns token IDs only."""

    def __init__(self, titok, flatten_output: bool = True):
        super().__init__()
        if titok.quantize_mode != "vq":
            raise ValueError(f"TiTokTokenEncoder only supports VQ models, got {titok.quantize_mode}.")

        self.encoder = titok.encoder
        self.quantize = titok.quantize
        self.flatten_output = flatten_output
        self.register_parameter("latent_tokens", titok.latent_tokens)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(pixel_values=pixel_values, latent_tokens=self.latent_tokens)
        _, result_dict = self.quantize(latent)
        tokens = result_dict["min_encoding_indices"]
        if self.flatten_output:
            tokens = tokens.reshape(tokens.shape[0], -1)
        return tokens
