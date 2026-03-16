import torch
import torch.nn as nn


class TiTokEncoderOnly(nn.Module):
    """Minimal TiTok inference wrapper that returns encoder latents only."""

    def __init__(self, titok):
        super().__init__()
        if titok.quantize_mode != "vq":
            raise ValueError(f"TiTokEncoderOnly only supports VQ models, got {titok.quantize_mode}.")

        self.encoder = titok.encoder
        self.register_parameter("latent_tokens", titok.latent_tokens)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values=pixel_values, latent_tokens=self.latent_tokens)


class TiTokVectorQuantizerTokens(nn.Module):
    """VQ-only wrapper that converts encoder latents into discrete token IDs."""

    def __init__(self, titok, flatten_output: bool = True):
        super().__init__()
        if titok.quantize_mode != "vq":
            raise ValueError(f"TiTokVectorQuantizerTokens only supports VQ models, got {titok.quantize_mode}.")

        self.quantize = titok.quantize
        self.flatten_output = flatten_output

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        _, result_dict = self.quantize(latent)
        tokens = result_dict["min_encoding_indices"]
        if self.flatten_output:
            tokens = tokens.reshape(tokens.shape[0], -1)
        return tokens


class TiTokTokenEncoder(nn.Module):
    """Minimal TiTok inference wrapper that returns token IDs only."""

    def __init__(self, titok, flatten_output: bool = True):
        super().__init__()
        self.encoder_only = TiTokEncoderOnly(titok)
        self.latents_to_tokens = TiTokVectorQuantizerTokens(titok, flatten_output=flatten_output)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        latent = self.encoder_only(pixel_values)
        return self.latents_to_tokens(latent)
