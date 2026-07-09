import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualAdd(nn.Module):
    def forward(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return residual + update


class QuantBoundary(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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


class TiTokEncoderPrefixSourceSdpaAttention(nn.Module):
    """Prefix encoder wrapper that forces source-level BHLD SDPA attention."""

    def __init__(self, titok, num_blocks: int):
        super().__init__()
        if titok.quantize_mode != "vq":
            raise ValueError(
                f"TiTokEncoderPrefixSourceSdpaAttention only supports VQ models, got {titok.quantize_mode}."
            )

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
            block = encoder.transformer[i]
            attn_output = block.attention_bhld_sdpa(block.ln_1(x))
            x = x + attn_output
            if block.mlp_ratio > 0:
                x = x + block.mlp(block.ln_2(x))
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


class TiTokEncoderPrefixSourceMatmulAttention(nn.Module):
    """Prefix encoder wrapper that forces source-level BHLD matmul attention."""

    def __init__(self, titok, num_blocks: int):
        super().__init__()
        if titok.quantize_mode != "vq":
            raise ValueError(
                f"TiTokEncoderPrefixSourceMatmulAttention only supports VQ models, got {titok.quantize_mode}."
            )

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
            block = encoder.transformer[i]
            attn_output = block.attention_bhld_matmul(block.ln_1(x))
            x = x + attn_output
            if block.mlp_ratio > 0:
                x = x + block.mlp(block.ln_2(x))
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


class TiTokEncoderOnlyReshapeBatch(nn.Module):
    """Full encoder wrapper that avoids unsqueeze by using reshape-based singleton batch dims."""

    def __init__(self, titok):
        super().__init__()
        if titok.quantize_mode != "vq":
            raise ValueError(f"TiTokEncoderOnlyReshapeBatch only supports VQ models, got {titok.quantize_mode}.")

        self.encoder = titok.encoder
        self.register_parameter("latent_tokens", titok.latent_tokens)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        batch_size = pixel_values.shape[0]

        x = encoder.patch_embed(pixel_values)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        class_embedding = encoder.class_embedding.reshape(1, encoder.class_embedding.shape[0], encoder.class_embedding.shape[1])
        class_embedding = class_embedding.expand(batch_size, -1, -1).to(x.dtype)
        x = torch.cat([class_embedding, x], dim=1)
        x = x + encoder.positional_embedding.to(x.dtype)

        latent_tokens = self.latent_tokens.reshape(1, self.latent_tokens.shape[0], self.latent_tokens.shape[1])
        latent_tokens = latent_tokens.expand(batch_size, -1, -1).to(x.dtype)
        latent_tokens = latent_tokens + encoder.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)

        x = encoder.ln_pre(x)
        x = x.permute(1, 0, 2)
        for block in encoder.transformer:
            x = block(x)
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


def _bmm_attention(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Equivalent attention path that expresses head mixing as rank-3 BMMs.

    The einsum attention variant uses einsum over rank-4 tensors:
      q: [B, L, H, D]
      k: [B, S, H, D]
      v: [B, S, H, D]

    ExecuTorch Arm lowering currently tends to rewrite that path into a
    higher-rank matmul bridge with unsqueeze/permute glue between delegate
    regions. Flattening heads into the batch dimension keeps the core attention
    math as rank-3 BMMs:
      [B*H, L, D] x [B*H, D, S] -> [B*H, L, S]

    That gives the lowering pipeline a simpler shape pattern to work with and
    is a better candidate for reducing CPU-side bridge ops.
    """

    seq_len, batch_size, embed_dim = x.shape
    num_heads = block.attn.num_heads
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5

    x_batch_first = x.transpose(0, 1)
    qkv = F.linear(
        x_batch_first,
        block.attn.in_proj_weight,
        block.attn.in_proj_bias,
    )
    q, k, v = qkv.chunk(3, dim=-1)

    q = q.reshape(batch_size, seq_len, num_heads, head_dim)
    k = k.reshape(batch_size, seq_len, num_heads, head_dim)
    v = v.reshape(batch_size, seq_len, num_heads, head_dim)

    q = q.permute(0, 2, 1, 3).reshape(batch_size * num_heads, seq_len, head_dim)
    k = k.permute(0, 2, 3, 1).reshape(batch_size * num_heads, head_dim, seq_len)
    v = v.permute(0, 2, 1, 3).reshape(batch_size * num_heads, seq_len, head_dim)

    attn_scores = torch.bmm(q, k) * scale
    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_output = torch.bmm(attn_probs, v)

    attn_output = attn_output.reshape(batch_size, num_heads, seq_len, head_dim)
    attn_output = attn_output.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, embed_dim
    )
    attn_output = F.linear(
        attn_output,
        block.attn.out_proj.weight,
        block.attn.out_proj.bias,
    )
    return attn_output.transpose(0, 1)


class TiTokEncoderOnlyBmmAttention(nn.Module):
    """Encoder-only wrapper that rewrites transformer attention into rank-3 BMMs."""

    def __init__(self, titok):
        super().__init__()
        if titok.quantize_mode != "vq":
            raise ValueError(
                f"TiTokEncoderOnlyBmmAttention only supports VQ models, got {titok.quantize_mode}."
            )

        self.encoder = titok.encoder
        self.register_parameter("latent_tokens", titok.latent_tokens)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        batch_size = pixel_values.shape[0]

        x = encoder.patch_embed(pixel_values)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                encoder.class_embedding.reshape(1, 1, encoder.width)
                .expand(batch_size, -1, -1)
                .to(x.dtype),
                x,
            ],
            dim=1,
        )
        x = x + encoder.positional_embedding.to(x.dtype)

        latent_tokens = self.latent_tokens.reshape(
            1, self.latent_tokens.shape[0], self.latent_tokens.shape[1]
        )
        latent_tokens = latent_tokens.expand(batch_size, -1, -1).to(x.dtype)
        latent_tokens = latent_tokens + encoder.latent_token_positional_embedding.to(
            x.dtype
        )
        x = torch.cat([x, latent_tokens], dim=1)

        x = encoder.ln_pre(x)
        x = x.permute(1, 0, 2)
        for block in encoder.transformer:
            attn_output = _bmm_attention(block, block.ln_1(x))
            x = x + attn_output
            if block.mlp_ratio > 0:
                x = x + block.mlp(block.ln_2(x))
        x = x.permute(1, 0, 2)

        latent_tokens = x[:, 1 + encoder.grid_size**2 :]
        latent_tokens = encoder.ln_post(latent_tokens)
        if encoder.is_legacy:
            latent_tokens = latent_tokens.reshape(
                batch_size, encoder.width, encoder.num_latent_tokens, 1
            )
        else:
            latent_tokens = latent_tokens.reshape(
                batch_size, encoder.num_latent_tokens, encoder.width, 1
            ).permute(0, 2, 1, 3)
        latent_tokens = encoder.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(
            batch_size, encoder.token_size, 1, encoder.num_latent_tokens
        )
        return latent_tokens


class TiTokEncoderOnlySourceMatmulAttention(nn.Module):
    """Encoder-only wrapper that uses the source-level BHLD matmul attention path."""

    def __init__(self, titok):
        super().__init__()
        if titok.quantize_mode != "vq":
            raise ValueError(
                f"TiTokEncoderOnlySourceMatmulAttention only supports VQ models, got {titok.quantize_mode}."
            )

        self.encoder = titok.encoder
        self.attn_residual_adds = nn.ModuleList(
            ResidualAdd() for _ in range(self.encoder.num_layers)
        )
        self.mlp_residual_adds = nn.ModuleList(
            ResidualAdd() for _ in range(self.encoder.num_layers)
        )
        for block in self.encoder.transformer:
            if not hasattr(block, "mlp_output_boundary"):
                block.mlp_output_boundary = QuantBoundary()
            if not hasattr(block, "post_gelu_boundary"):
                block.post_gelu_boundary = QuantBoundary()
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
        for block_index, block in enumerate(encoder.transformer):
            attn_output = block.attention_bhld_matmul(block.ln_1(x))
            x = self.attn_residual_adds[block_index](x, attn_output)
            if block.mlp_ratio > 0:
                mlp_output = block.mlp.c_fc(block.ln_2(x))
                mlp_output = block.mlp.gelu(mlp_output)
                mlp_output = block.post_gelu_boundary(mlp_output)
                mlp_output = block.mlp.c_proj(mlp_output)
                mlp_output = block.mlp_output_boundary(mlp_output)
                x = self.mlp_residual_adds[block_index](x, mlp_output)
        x = x.permute(1, 0, 2)

        latent_tokens = x[:, 1 + encoder.grid_size ** 2 :]
        latent_tokens = encoder.ln_post(latent_tokens)
        if encoder.is_legacy:
            latent_tokens = latent_tokens.reshape(
                batch_size, encoder.width, encoder.num_latent_tokens, 1
            )
        else:
            latent_tokens = latent_tokens.reshape(
                batch_size, encoder.num_latent_tokens, encoder.width, 1
            ).permute(0, 2, 1, 3)
        latent_tokens = encoder.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(
            batch_size, encoder.token_size, 1, encoder.num_latent_tokens
        )
        return latent_tokens


class TiTokEncoderOnlySourceQueryChunkedMatmulAttention(nn.Module):
    """Encoder-only wrapper that chunks BHLD matmul attention over query tokens."""

    def __init__(self, titok, query_chunk_size: int = 128):
        super().__init__()
        if titok.quantize_mode != "vq":
            raise ValueError(
                "TiTokEncoderOnlySourceQueryChunkedMatmulAttention only supports "
                f"VQ models, got {titok.quantize_mode}."
            )

        self.encoder = titok.encoder
        self.query_chunk_size = query_chunk_size
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
        for block in encoder.transformer:
            attn_output = block.attention_bhld_query_chunked_matmul(
                block.ln_1(x),
                self.query_chunk_size,
            )
            x = x + attn_output
            if block.mlp_ratio > 0:
                x = x + block.mlp(block.ln_2(x))
        x = x.permute(1, 0, 2)

        latent_tokens = x[:, 1 + encoder.grid_size ** 2 :]
        latent_tokens = encoder.ln_post(latent_tokens)
        if encoder.is_legacy:
            latent_tokens = latent_tokens.reshape(
                batch_size, encoder.width, encoder.num_latent_tokens, 1
            )
        else:
            latent_tokens = latent_tokens.reshape(
                batch_size, encoder.num_latent_tokens, encoder.width, 1
            ).permute(0, 2, 1, 3)
        latent_tokens = encoder.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(
            batch_size, encoder.token_size, 1, encoder.num_latent_tokens
        )
        return latent_tokens


class TiTokEncoderOnlyEinsumAttention(nn.Module):
    """Encoder-only wrapper that uses the source-level einsum attention path."""

    def __init__(self, titok):
        super().__init__()
        if titok.quantize_mode != "vq":
            raise ValueError(
                f"TiTokEncoderOnlyEinsumAttention only supports VQ models, got {titok.quantize_mode}."
            )

        self.encoder = titok.encoder
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
        for block in encoder.transformer:
            attn_output = block.attention_einsum(block.ln_1(x))
            x = x + attn_output
            if block.mlp_ratio > 0:
                x = x + block.mlp(block.ln_2(x))
        x = x.permute(1, 0, 2)

        latent_tokens = x[:, 1 + encoder.grid_size ** 2 :]
        latent_tokens = encoder.ln_post(latent_tokens)
        if encoder.is_legacy:
            latent_tokens = latent_tokens.reshape(
                batch_size, encoder.width, encoder.num_latent_tokens, 1
            )
        else:
            latent_tokens = latent_tokens.reshape(
                batch_size, encoder.num_latent_tokens, encoder.width, 1
            ).permute(0, 2, 1, 3)
        latent_tokens = encoder.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(
            batch_size, encoder.token_size, 1, encoder.num_latent_tokens
        )
        return latent_tokens


class TiTokEncoderOnlySourceSdpaAttention(nn.Module):
    """Encoder-only wrapper that uses the source-level BHLD SDPA attention path."""

    def __init__(self, titok):
        super().__init__()
        if titok.quantize_mode != "vq":
            raise ValueError(
                f"TiTokEncoderOnlySourceSdpaAttention only supports VQ models, got {titok.quantize_mode}."
            )

        self.encoder = titok.encoder
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
        for block in encoder.transformer:
            attn_output = block.attention_bhld_sdpa(block.ln_1(x))
            x = x + attn_output
            if block.mlp_ratio > 0:
                x = x + block.mlp(block.ln_2(x))
        x = x.permute(1, 0, 2)

        latent_tokens = x[:, 1 + encoder.grid_size ** 2 :]
        latent_tokens = encoder.ln_post(latent_tokens)
        if encoder.is_legacy:
            latent_tokens = latent_tokens.reshape(
                batch_size, encoder.width, encoder.num_latent_tokens, 1
            )
        else:
            latent_tokens = latent_tokens.reshape(
                batch_size, encoder.num_latent_tokens, encoder.width, 1
            ).permute(0, 2, 1, 3)
        latent_tokens = encoder.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(
            batch_size, encoder.token_size, 1, encoder.num_latent_tokens
        )
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


class TiTokTokenEncoderFromModules(nn.Module):
    """Compose an arbitrary encoder-only wrapper with the float VQ tokenizer."""

    def __init__(self, encoder_only: nn.Module, latents_to_tokens: nn.Module):
        super().__init__()
        self.encoder_only = encoder_only
        self.latents_to_tokens = latents_to_tokens

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        latent = self.encoder_only(pixel_values)
        return self.latents_to_tokens(latent)
