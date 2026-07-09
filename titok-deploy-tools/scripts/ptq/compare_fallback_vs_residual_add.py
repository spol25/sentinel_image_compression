#!/usr/bin/env python3
"""Compare quantized TiTok reconstructions with a quantized fallback checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from PIL import Image, ImageDraw, ImageFont
from einops import rearrange

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.ptq_tools.ptq import (
    build_encoder_quantizer_split,
    calibrate_prepared_encoder,
    convert_encoder_after_ptq,
    export_encoder_program,
    load_manifest_records,
    prepare_exported_encoder_for_ptq,
)
from titok_deploy_tools.wrapper_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.wrapper_tools.utils import load_image, resolve_input_path, resolve_output_dir, save_reconstruction


def psnr(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    mse = torch.mean((reference - candidate) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def ssim_simple(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    x = reference.detach().to(torch.float64)
    y = candidate.detach().to(torch.float64)
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


def reconstruction_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    diff = (candidate - reference).abs()
    return {
        "psnr": psnr(reference, candidate),
        "ssim": ssim_simple(reference, candidate),
        "mae": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((candidate - reference) ** 2)).item()),
    }


def summarize(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    keys = [
        key
        for key, value in rows[0].items()
        if key.startswith(prefix) and isinstance(value, int | float)
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def latent_to_tokens(titok, latent: torch.Tensor) -> torch.Tensor:
    z = latent.detach().to("cpu", dtype=torch.float32).permute(0, 2, 3, 1).contiguous()
    flat = z.reshape(-1, z.shape[-1])
    if titok.quantize.use_l2_norm:
        flat = torch.nn.functional.normalize(flat, dim=-1)
        embedding = torch.nn.functional.normalize(titok.quantize.embedding.weight.detach().to("cpu"), dim=-1)
    else:
        embedding = titok.quantize.embedding.weight.detach().to("cpu")
    distances = (
        torch.sum(flat**2, dim=1, keepdim=True)
        + torch.sum(embedding**2, dim=1)
        - 2 * torch.matmul(flat, embedding.t())
    )
    return torch.argmin(distances, dim=1).reshape(latent.shape[0], latent.shape[2], latent.shape[3])


def decode_titok_tokens(titok, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.clamp(titok.decode_tokens(tokens.reshape(tokens.shape[0], 1, -1)), 0.0, 1.0)


def build_quantized_titok_encoder_module(
    encoder_only: nn.Module,
    example_input: torch.Tensor,
    calibration_images: list[Path],
    image_size: int,
    quantization_profile: str,
) -> torch.fx.GraphModule:
    exported_program = export_encoder_program(encoder_only, example_input)
    prepared_encoder, _ = prepare_exported_encoder_for_ptq(
        exported_program,
        backend="ethosu",
        is_per_channel=True,
        quantization_profile=quantization_profile,
        ethos_target="ethos-u65-256",
        ethos_system_config="Ethos_U65_High_End",
        ethos_memory_mode="Dedicated_Sram_384KB",
        ethos_config_ini="Arm/vela.ini",
        ethos_extra_flags=[],
        quantize_matmul=False,
    )
    calibrate_prepared_encoder(prepared_encoder, calibration_images, image_size)
    quantized_encoder = convert_encoder_after_ptq(prepared_encoder, backend="ethosu").to("cpu")
    return torch.export.export(quantized_encoder, (example_input,), strict=True).module().to("cpu")


class DistillEncTiTokDec(nn.Module):
    def __init__(self, student_model: nn.Module, teacher_model: nn.Module):
        super().__init__()
        self.student_model = student_model
        self.quant = teacher_model.quantize
        self.decoder = teacher_model.decoder
        self.pixel_quant = nn.Parameter(teacher_model.pixel_quantize.embedding.weight.detach().clone())
        self.pixel_decoder = teacher_model.pixel_decoder

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.student_model(image)
        tokens = torch.argmax(logits, dim=1)
        codebook_weights = self.quant.embedding.weight
        z_quantized = codebook_weights[tokens.reshape(-1)].reshape(tokens.shape[0], tokens.shape[1], -1)
        z_quantized = z_quantized.view(tokens.shape[0], 1, tokens.shape[1], -1)
        z_quantized = rearrange(z_quantized, "b h w c -> b c h w").contiguous()
        z_quantized, _ = self.quant(z_quantized)
        decoded = self.decoder(z_quantized)
        quantized_states = torch.einsum("nchw,cd->ndhw", decoded.softmax(1), self.pixel_quant)
        decoded = torch.clamp(self.pixel_decoder(quantized_states), 0.0, 1.0)
        return decoded, tokens


def load_fallback_model(distill_repo_root: Path, titok, checkpoint_path: Path) -> nn.Module:
    prod_root = distill_repo_root / "TiTok-Distill-Prod" / "titok-distill-prod"
    one_d_tokenizer = prod_root / "1d-tokenizer"
    for path in (prod_root, one_d_tokenizer):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    from Encoders import GeminiV0Encoder

    student = GeminiV0Encoder(
        arch="101",
        pretrained=False,
        codebook_size=4096,
        latent_dim=128,
        freeze_backbone=False,
        dropout=0.1,
    )
    model = DistillEncTiTokDec(student, titok)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if any(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Fallback checkpoint mismatch. Missing={missing}, unexpected={unexpected}")
    return model.eval().to("cpu")


def quantize_fallback_student(
    fallback_model: DistillEncTiTokDec,
    calibration_images: list[Path],
    image_size: int,
) -> DistillEncTiTokDec:
    torch.backends.quantized.engine = "qnnpack"
    example_input = load_image(calibration_images[0], image_size).to("cpu")
    student = fallback_model.student_model.eval().to("cpu")
    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    prepared_student = prepare_fx(student, qconfig_mapping, (example_input,))
    with torch.no_grad():
        for image_path in calibration_images:
            prepared_student(load_image(image_path, image_size).to("cpu"))
    fallback_model.student_model = convert_fx(prepared_student).eval().to("cpu")
    return fallback_model.eval().to("cpu")


def load_calibration_images(path: Path, count: int) -> tuple[list[Path], Path]:
    if path.is_dir():
        images = sorted(
            image_path
            for image_path in path.iterdir()
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    else:
        images = load_manifest_records(path)
    existing = [image_path for image_path in images if image_path.exists()]
    if len(existing) < count:
        raise RuntimeError(
            f"Need {count} readable calibration images from {path}, found {len(existing)}. "
            "Check mounted-volume permissions and paths."
        )
    return existing[:count], path


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor[0].detach().to("cpu", dtype=torch.float32).clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).astype(np.uint8))


def make_panel(path: Path, title_values: list[tuple[str, torch.Tensor]]) -> None:
    images = [tensor_to_image(tensor) for _, tensor in title_values]
    width, height = images[0].size
    label_h = 34
    canvas = Image.new("RGB", (width * len(images), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, ((label, _), image) in enumerate(zip(title_values, images)):
        x = index * width
        canvas.paste(image, (x, label_h))
        draw.text((x + 8, 10), label, fill=(0, 0, 0), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--titok-root", default="/Users/sruthipolali/Documents/Playground/1d-tokenizer", type=Path)
    parser.add_argument("--repo-id", default="yucornetto/tokenizer_titok_s128_imagenet")
    parser.add_argument("--distill-repo-root", default="/Users/sruthipolali/Documents/Playground/sentinel-titok-distill", type=Path)
    parser.add_argument(
        "--fallback-checkpoint",
        default=REPO_ROOT / "outputs/Fallback solution/1MSELoss_0.01perceptualloss_0.1gradloss_0.01ssimloss_1fourierloss_0.0075ganloss_SSIM_best.pt",
        type=Path,
    )
    parser.add_argument("--image-dir", default=REPO_ROOT / "outputs/better_inputs", type=Path)
    parser.add_argument(
        "--calibration-manifest",
        default=Path("/Volumes/Media/snapshot_serengeti_balanced_ptq_dataset/manifests/calibration_manifest.json"),
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=REPO_ROOT / "outputs/better_inputs_quantized_fallback_vs_residual_add_full_a16w8_calib500",
        type=Path,
    )
    parser.add_argument("--calibration-count", default=500, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_titok_root_to_path(str(args.titok_root))
    from modeling.titok import TiTok

    output_dir = resolve_output_dir(REPO_ROOT, str(args.output_dir))
    image_paths = sorted(
        path for path in args.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
    titok.requires_grad_(False)
    image_size = int(titok.config.dataset.preprocessing.crop_size)

    encoder_only, _, _ = build_encoder_quantizer_split(titok, encoder_variant="source_matmul_attention")
    encoder_only = encoder_only.eval().to("cpu")
    encoder_only.requires_grad_(False)

    calibration_path = resolve_input_path(str(args.calibration_manifest), REPO_ROOT)
    calibration_images, calibration_source = load_calibration_images(calibration_path, args.calibration_count)
    example_input = load_image(calibration_images[0], image_size).to("cpu")
    residual_module = build_quantized_titok_encoder_module(
        encoder_only,
        example_input,
        calibration_images,
        image_size,
        "int8_surface_transformer_norm_residual_a16w8",
    )
    full_a16w8_module = build_quantized_titok_encoder_module(
        encoder_only,
        example_input,
        calibration_images,
        image_size,
        "a16w8",
    )

    fallback_model = load_fallback_model(args.distill_repo_root, titok, args.fallback_checkpoint)
    fallback_model = quantize_fallback_student(fallback_model, calibration_images, image_size)

    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths):
        image = load_image(image_path, image_size).to("cpu")
        with torch.no_grad():
            residual_latent = residual_module(image).detach().to("cpu", dtype=torch.float32)
            residual_tokens = latent_to_tokens(titok, residual_latent)
            residual_recon = decode_titok_tokens(titok, residual_tokens)
            full_a16w8_latent = full_a16w8_module(image).detach().to("cpu", dtype=torch.float32)
            full_a16w8_tokens = latent_to_tokens(titok, full_a16w8_latent)
            full_a16w8_recon = decode_titok_tokens(titok, full_a16w8_tokens)
            fallback_recon, fallback_tokens = fallback_model(image)
        stem = f"{index:03d}_{image_path.stem}"
        residual_path = output_dir / "reconstructions" / f"{stem}_residual_add_a16w8.png"
        full_a16w8_path = output_dir / "reconstructions" / f"{stem}_full_a16w8.png"
        fallback_path = output_dir / "reconstructions" / f"{stem}_fallback_ssim_best_int8.png"
        reference_path = output_dir / "references" / f"{stem}_reference.png"
        panel_path = output_dir / "panels" / f"{stem}_reference_residual_fulla16w8_fallback.png"
        save_reconstruction(image, reference_path)
        save_reconstruction(residual_recon, residual_path)
        save_reconstruction(full_a16w8_recon, full_a16w8_path)
        save_reconstruction(fallback_recon, fallback_path)
        make_panel(
            panel_path,
            [
                ("reference", image),
                ("residual-add-a16w8", residual_recon),
                ("full-a16w8", full_a16w8_recon),
                ("fallback SSIM best INT8", fallback_recon),
            ],
        )
        residual_metrics = reconstruction_metrics(image, residual_recon)
        full_a16w8_metrics = reconstruction_metrics(image, full_a16w8_recon)
        fallback_metrics = reconstruction_metrics(image, fallback_recon)
        token_exact = float((residual_tokens.reshape(-1) == fallback_tokens.reshape(-1).to("cpu")).to(torch.float32).mean().item())
        full_a16w8_fallback_token_exact = float((full_a16w8_tokens.reshape(-1) == fallback_tokens.reshape(-1).to("cpu")).to(torch.float32).mean().item())
        residual_full_a16w8_token_exact = float((residual_tokens.reshape(-1) == full_a16w8_tokens.reshape(-1)).to(torch.float32).mean().item())
        rows.append(
            {
                "index": index,
                "image": str(image_path),
                "stem": stem,
                "residual_psnr": residual_metrics["psnr"],
                "residual_ssim": residual_metrics["ssim"],
                "residual_mae": residual_metrics["mae"],
                "residual_rmse": residual_metrics["rmse"],
                "full_a16w8_psnr": full_a16w8_metrics["psnr"],
                "full_a16w8_ssim": full_a16w8_metrics["ssim"],
                "full_a16w8_mae": full_a16w8_metrics["mae"],
                "full_a16w8_rmse": full_a16w8_metrics["rmse"],
                "fallback_psnr": fallback_metrics["psnr"],
                "fallback_ssim": fallback_metrics["ssim"],
                "fallback_mae": fallback_metrics["mae"],
                "fallback_rmse": fallback_metrics["rmse"],
                "fallback_minus_residual_psnr": fallback_metrics["psnr"] - residual_metrics["psnr"],
                "fallback_minus_residual_ssim": fallback_metrics["ssim"] - residual_metrics["ssim"],
                "fallback_minus_full_a16w8_psnr": fallback_metrics["psnr"] - full_a16w8_metrics["psnr"],
                "fallback_minus_full_a16w8_ssim": fallback_metrics["ssim"] - full_a16w8_metrics["ssim"],
                "full_a16w8_minus_residual_psnr": full_a16w8_metrics["psnr"] - residual_metrics["psnr"],
                "full_a16w8_minus_residual_ssim": full_a16w8_metrics["ssim"] - residual_metrics["ssim"],
                "residual_fallback_token_exact_agreement": token_exact,
                "full_a16w8_fallback_token_exact_agreement": full_a16w8_fallback_token_exact,
                "residual_full_a16w8_token_exact_agreement": residual_full_a16w8_token_exact,
                "reference_png": str(reference_path),
                "residual_png": str(residual_path),
                "full_a16w8_png": str(full_a16w8_path),
                "fallback_png": str(fallback_path),
                "panel_png": str(panel_path),
            }
        )

    summary = {
        "status": "succeeded",
        "image_dir": str(args.image_dir),
        "image_count": len(image_paths),
        "calibration_manifest": str(calibration_source),
        "calibration_count": len(calibration_images),
        "fallback_checkpoint": str(args.fallback_checkpoint),
        "fallback_quantization": "torch_fx_static_int8_qnnpack_student_encoder",
        "residual_profile": "int8_surface_transformer_norm_residual_a16w8",
        "full_a16w8_profile": "a16w8",
        "titok_encoder_variant": "source_matmul_attention",
        "mean": {
            **summarize(rows, "residual_"),
            **summarize(rows, "full_a16w8_"),
            **summarize(rows, "fallback_"),
        },
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["mean"], indent=2))
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
