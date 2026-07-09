#!/usr/bin/env python3
"""Run fallback float encoder inference using the distill repo export/decode path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISTILL_ROOT = Path("/Users/sruthipolali/Documents/Playground/sentinel-titok-distill")
DEFAULT_TITOK_DEPLOY_ROOT = REPO_ROOT
DEFAULT_FALLBACK_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "_archive"
    / "2026-06-fallback-visual-evals"
    / "fallback_solution"
    / "1MSELoss_0.01perceptualloss_0.1gradloss_0.01ssimloss_1fourierloss_0.0075ganloss_SSIM_best.pt"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "repo_exact_fallback_float"
DEFAULT_CASES = (
    (
        "image 1",
        REPO_ROOT / "outputs" / "better_inputs" / "0026_day_near_S2_H08_R3_IMAG0666.jpg",
        REPO_ROOT / "outputs" / "titok_vs_fallback_step_by_step_comparison" / "image 1",
    ),
    (
        "image-2",
        REPO_ROOT / "outputs" / "better_inputs" / "0030_day_near_S3_I08_R12_IMAG0248.jpg",
        REPO_ROOT / "outputs" / "titok_vs_fallback_step_by_step_comparison" / "image-2",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distill-root", type=Path, default=DEFAULT_DISTILL_ROOT)
    parser.add_argument("--fallback-checkpoint", type=Path, default=DEFAULT_FALLBACK_CHECKPOINT)
    parser.add_argument("--repo-id", default="yucornetto/tokenizer_titok_s128_imagenet")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force-export", action="store_true")
    return parser.parse_args()


def add_repo_paths(distill_root: Path) -> tuple[Path, Path, Path]:
    prod_root = distill_root / "TiTok-Distill-Prod" / "titok-distill-prod"
    tokenizer_root = prod_root / "1d-tokenizer"
    on_device_root = prod_root / "on-device"
    for path in (prod_root, tokenizer_root, on_device_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return prod_root, tokenizer_root, on_device_root


class DistillEncTiTokDec(nn.Module):
    """Repo-compatible state-dict shell for the final fallback checkpoint."""

    def __init__(self, student_model: nn.Module, teacher_model: nn.Module):
        super().__init__()
        self.student_model = student_model
        self.quant = teacher_model.quantize
        self.decoder = teacher_model.decoder
        self.pixel_quant = nn.Parameter(teacher_model.pixel_quantize.embedding.weight.detach().clone())
        self.pixel_decoder = teacher_model.pixel_decoder

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        z = self.student_model(img)
        z_one_hot = torch.nn.functional.gumbel_softmax(z, tau=1.0, hard=True, dim=1)
        codebook_weights = self.quant.embedding.weight
        z_quantized = torch.matmul(z_one_hot.permute(0, 2, 1), codebook_weights)
        batch, seq_len, dim = z_quantized.shape
        z_quantized = z_quantized.view(batch, 1, seq_len, dim)
        z_quantized = rearrange(z_quantized, "b h w c -> b c h w").contiguous()
        z_quantized, _ = self.quant(z_quantized)
        x_hat = self.decoder(z_quantized)
        quantized_states = torch.einsum("nchw,cd->ndhw", x_hat.softmax(1), self.pixel_quant)
        x_hat = self.pixel_decoder(quantized_states)
        return torch.clamp(x_hat, 0.0, 1.0)


class STQuantizerDecoder(nn.Module):
    """Same token-to-image decoder structure used by export_v3/finetune2onnx.py."""

    def __init__(self, st_model: DistillEncTiTokDec, titok_model):
        super().__init__()
        self.quant = st_model.quant
        self.decoder = st_model.decoder
        self.pixel_quant = st_model.pixel_quant
        self.pixel_decoder = getattr(titok_model, "pixel_decoder", None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch, seq_len = z.shape
        z_quantized = self.quant.get_codebook_entry(z.reshape(-1)).reshape(batch, 1, seq_len, -1)
        z_quantized = rearrange(z_quantized, "b h w c -> b c h w").contiguous()
        x_hat = self.decoder(z_quantized)
        quantized_states = torch.einsum("nchw,cd->ndhw", x_hat.softmax(1), self.pixel_quant)
        x_hat = self.pixel_decoder(quantized_states)
        return torch.clamp(x_hat, 0.0, 1.0)


def load_repo_model(args: argparse.Namespace) -> tuple[nn.Module, nn.Module, int]:
    from Encoders import GeminiV0Encoder
    from modeling.titok import TiTok

    titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
    titok.requires_grad_(False)
    image_size = int(titok.config.dataset.preprocessing.crop_size)
    student = GeminiV0Encoder(
        arch="101",
        pretrained=False,
        codebook_size=4096,
        latent_dim=128,
        freeze_backbone=False,
        dropout=0.1,
    )
    st_model = DistillEncTiTokDec(student, titok)
    checkpoint = torch.load(args.fallback_checkpoint, map_location="cpu")
    if any(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    missing, unexpected = st_model.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Fallback checkpoint mismatch. Missing={missing}, unexpected={unexpected}")
    st_model.eval().to("cpu")
    st_model.requires_grad_(False)
    quant_dec = STQuantizerDecoder(st_model, titok).eval().to("cpu")
    quant_dec.requires_grad_(False)
    return st_model.student_model.eval().to("cpu"), quant_dec, image_size


def export_artifacts(
    encoder: nn.Module,
    quant_dec: nn.Module,
    image_size: int,
    output_root: Path,
    force: bool,
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    encoder_onnx = output_root / "encoder.onnx"
    decoder_jit = output_root / "quant_dec_jit.pt"
    if force or not encoder_onnx.exists():
        example_image = torch.randn(1, 3, image_size, image_size)
        torch.onnx.export(
            encoder,
            example_image,
            encoder_onnx,
            export_params=True,
            input_names=["input"],
            output_names=["output"],
            opset_version=20,
        )
    if force or not decoder_jit.exists():
        example_tokens = torch.randint(0, 4096, (1, 128), dtype=torch.long)
        traced_model = torch.jit.trace(quant_dec, example_tokens)
        traced_model.save(str(decoder_jit))
    return encoder_onnx, decoder_jit


def repo_float_preprocess(image_path: Path, image_size: int) -> np.ndarray:
    from utils import pad_image

    image = Image.open(image_path).convert("RGB")
    image = pad_image(image).resize((image_size, image_size))
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(image_np, (2, 0, 1))[None].astype(np.float32)


def save_recon_array(image_array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(image_array, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def run_case(
    *,
    case_name: str,
    image_path: Path,
    case_dir: Path,
    encoder_onnx: Path,
    decoder_jit: Path,
    image_size: int,
) -> dict:
    from utils import ByteReader, ByteWriter, postprocess_v1

    case_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(encoder_onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    preprocessed = repo_float_preprocess(image_path, image_size)
    logits = session.run(None, {input_name: preprocessed})[0]
    tokens = np.argmax(logits, axis=1).astype(np.int64)

    tokens_bin = case_dir / "repo_exact_float_encoder_tokens.bin"
    tokens_json = case_dir / "repo_exact_float_encoder_tokens.json"
    ByteWriter()(tokens.squeeze().astype(np.uint16), str(tokens_bin))
    roundtrip_tokens = ByteReader()(str(tokens_bin)).astype(np.int64)
    tokens_json.write_text(
        json.dumps(
            {
                "shape": [1, int(roundtrip_tokens.shape[0])],
                "tokens": roundtrip_tokens.reshape(1, -1).tolist(),
                "source": "repo ByteWriter/ByteReader roundtrip from float encoder ONNX logits",
            },
            indent=2,
        )
        + "\n"
    )

    decoder = torch.jit.load(str(decoder_jit), map_location="cpu").eval()
    token_tensor = torch.from_numpy(roundtrip_tokens).to(torch.long).view(1, -1)
    with torch.no_grad():
        raw_recon = torch.clamp(decoder(token_tensor), 0.0, 1.0)
        postprocessed = postprocess_v1(raw_recon)

    raw_recon_array = raw_recon[0].permute(1, 2, 0).detach().cpu().numpy()
    raw_recon_path = case_dir / "repo_exact_float_encoder_reconstruction_raw.png"
    post_recon_path = case_dir / "repo_exact_float_encoder_reconstruction.png"
    save_recon_array(raw_recon_array, raw_recon_path)
    save_recon_array(postprocessed, post_recon_path)
    return {
        "case": case_name,
        "input_image": str(image_path),
        "encoder_onnx": str(encoder_onnx),
        "decoder_jit": str(decoder_jit),
        "tokens_bin": str(tokens_bin),
        "tokens_json": str(tokens_json),
        "raw_reconstruction": str(raw_recon_path),
        "postprocessed_reconstruction": str(post_recon_path),
        "logits_shape": list(logits.shape),
    }


def main() -> None:
    args = parse_args()
    add_repo_paths(args.distill_root)
    encoder, quant_dec, image_size = load_repo_model(args)
    encoder_onnx, decoder_jit = export_artifacts(
        encoder,
        quant_dec,
        image_size,
        args.output_root,
        args.force_export,
    )
    runs = [
        run_case(
            case_name=case_name,
            image_path=image_path,
            case_dir=case_dir,
            encoder_onnx=encoder_onnx,
            decoder_jit=decoder_jit,
            image_size=image_size,
        )
        for case_name, image_path, case_dir in DEFAULT_CASES
    ]
    metadata = {
        "status": "succeeded",
        "path": "repo-style float encoder ONNX -> argmax(axis=1) -> ByteWriter/ByteReader -> quant_dec_jit.pt -> postprocess_v1",
        "fallback_checkpoint": str(args.fallback_checkpoint),
        "image_size": image_size,
        "runs": runs,
    }
    metadata_path = args.output_root / "repo_exact_float_run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
