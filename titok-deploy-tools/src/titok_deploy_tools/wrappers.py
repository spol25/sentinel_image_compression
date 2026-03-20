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


class TiTokEncoderPrefix(nn.Module):
    """Encoder-only wrapper that runs the TiTok encoder up to a chosen block depth."""

    def __init__(self, titok, num_blocks: int):
        super().__init__()
        if titok.quantize_mode != "vq":
            raise ValueError(f"TiTokEncoderPrefix only supports VQ models, got {titok.quantize_mode}.")

        encoder = titok.encoder
        if num_blocks < 0 or num_blocks > encoder.num_layers:
            raise ValueError(f"num_blocks must be in [0, {encoder.num_layers}], got {num_blocks}.")

        self.encoder = encoder
        self.num_blocks = num_blocks
        self.register_parameter("latent_tokens", titok.latent_tokens)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        batch_size = pixel_values.shape[0]

        x = encoder.patch_embed(pixel_values)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                encoder.class_embedding.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype),
                x,
            ],
            dim=1,
        )
        x = x + encoder.positional_embedding.to(x.dtype)

        latent_tokens = self.latent_tokens.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        latent_tokens = latent_tokens + encoder.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)

        x = encoder.ln_pre(x)
        x = x.permute(1, 0, 2)
        for i in range(self.num_blocks):
            x = encoder.transformer[i](x)
        x = x.permute(1, 0, 2)

        latent_tokens = x[:, 1 + encoder.grid_size ** 2 :]
        latent_tokens = encoder.ln_post(latent_tokens)
        if encoder.is_legacy:
            latent_tokens = latent_tokens.reshape(batch_size, encoder.width, encoder.num_latent_tokens, 1)
        else:
            latent_tokens = latent_tokens.reshape(
                batch_size, encoder.num_latent_tokens, encoder.width, 1
            ).permute(0, 2, 1, 3)
        latent_tokens = encoder.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(batch_size, encoder.token_size, 1, encoder.num_latent_tokens)
        return latent_tokens


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
