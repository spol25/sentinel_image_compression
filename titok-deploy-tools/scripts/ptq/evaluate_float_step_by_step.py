#!/usr/bin/env python3
"""Generate float source-matmul and original fallback comparison outputs for one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_fallback_vs_residual_add import (  # noqa: E402
    decode_titok_tokens,
    load_fallback_model,
)
from titok_deploy_tools.ptq_tools.ptq import (  # noqa: E402
    build_encoder_quantizer_split,
    compare_latent_tensors,
)
from titok_deploy_tools.wrapper_tools.titok_env import add_titok_root_to_path  # noqa: E402
from titok_deploy_tools.wrapper_tools.utils import load_image, save_reconstruction  # noqa: E402


DEFAULT_COMPARISON_DIR = REPO_ROOT / "outputs" / "titok_vs_fallback_step_by_step_comparison"
DEFAULT_INPUT = REPO_ROOT / "outputs" / "better_inputs" / "0026_day_near_S2_H08_R3_IMAG0666.jpg"
DEFAULT_FALLBACK_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "_archive"
    / "2026-06-fallback-visual-evals"
    / "fallback_solution"
    / "1MSELoss_0.01perceptualloss_0.1gradloss_0.01ssimloss_1fourierloss_0.0075ganloss_SSIM_best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--titok-root", default="/Users/sruthipolali/Documents/Playground/1d-tokenizer", type=Path)
    parser.add_argument("--repo-id", default="yucornetto/tokenizer_titok_s128_imagenet")
    parser.add_argument("--distill-repo-root", default="/Users/sruthipolali/Documents/Playground/sentinel-titok-distill", type=Path)
    parser.add_argument("--input-image", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_COMPARISON_DIR, type=Path)
    parser.add_argument("--fallback-checkpoint", default=DEFAULT_FALLBACK_CHECKPOINT, type=Path)
    return parser.parse_args()


def save_tokens(tokens: torch.Tensor, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_tokens = tokens.detach().to("cpu", dtype=torch.long).reshape(tokens.shape[0], -1)
    path.write_text(
        json.dumps(
            {
                "shape": list(flat_tokens.shape),
                "tokens": flat_tokens.tolist(),
                "metadata": metadata,
            },
            indent=2,
        )
        + "\n"
    )


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor[0]
        .detach()
        .to("cpu", dtype=torch.float32)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray((array * 255.0).astype("uint8"))


def make_stitched_panel(path: Path, panels: list[tuple[str, torch.Tensor]]) -> None:
    images = [tensor_to_image(tensor) for _, tensor in panels]
    width, height = images[0].size
    label_h = 34
    canvas = Image.new("RGB", (width * len(images), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, ((label, _), image) in enumerate(zip(panels, images)):
        x = index * width
        canvas.paste(image, (x, label_h))
        draw.text((x + 8, 10), label, fill=(0, 0, 0), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = parse_args()
    add_titok_root_to_path(str(args.titok_root))
    from modeling.titok import TiTok

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
    titok.requires_grad_(False)
    image_size = int(titok.config.dataset.preprocessing.crop_size)
    image = load_image(args.input_image, image_size).to("cpu")

    baseline_encoder, _, default_wrapper = build_encoder_quantizer_split(titok, encoder_variant="baseline")
    source_encoder, _, source_wrapper = build_encoder_quantizer_split(
        titok,
        encoder_variant="source_matmul_attention",
    )
    for module in (baseline_encoder, default_wrapper, source_encoder, source_wrapper):
        module.eval().to("cpu")
        module.requires_grad_(False)

    with torch.no_grad():
        reference_latent = baseline_encoder(image).detach().to("cpu", dtype=torch.float32)
        default_tokens = default_wrapper(image).detach().to("cpu")
        default_recon = decode_titok_tokens(titok, default_tokens.reshape(default_tokens.shape[0], -1))
        source_latent = source_encoder(image).detach().to("cpu", dtype=torch.float32)
        source_tokens = source_wrapper(image).detach().to("cpu")
        source_recon = decode_titok_tokens(titok, source_tokens.reshape(source_tokens.shape[0], -1))

    token_metadata = {
        "repo_id": args.repo_id,
        "input_image": str(args.input_image),
        "image_size": image_size,
    }
    save_tokens(default_tokens, output_dir / "default_titok_token_encoder_tokens.json", token_metadata)
    save_reconstruction(default_recon, output_dir / "default_titok_token_encoder_reconstruction.png")
    save_tokens(source_tokens, output_dir / "source_matmul_token_encoder_tokens.json", token_metadata)
    save_reconstruction(source_recon, output_dir / "source_matmul_token_encoder_reconstruction.png")

    stem = args.input_image.stem
    save_tokens(default_tokens, output_dir / f"{stem}_tokens.json", token_metadata)
    save_reconstruction(default_recon, output_dir / f"{stem}_reconstruction.png")
    save_tokens(default_tokens, output_dir / "reconstruct_titok_example_s128_tokens.json", token_metadata)
    save_reconstruction(default_recon, output_dir / "reconstruct_titok_example_s128_reconstruction.png")
    save_reconstruction(default_recon, output_dir / "reconstruct_titok_example_s128_reconstruction_from_tokens.png")

    fallback_model = load_fallback_model(args.distill_repo_root, titok, args.fallback_checkpoint)
    fallback_model.eval().to("cpu")
    fallback_model.requires_grad_(False)
    with torch.no_grad():
        fallback_recon, fallback_tokens = fallback_model(image)
    fallback_metadata = {
        **token_metadata,
        "checkpoint": str(args.fallback_checkpoint),
        "model": "original_non_quantized_fallback",
    }
    save_tokens(fallback_tokens, output_dir / "fallback_original_non_quantized_tokens.json", fallback_metadata)
    save_reconstruction(fallback_recon, output_dir / "fallback_original_non_quantized_reconstruction.png")

    make_stitched_panel(
        output_dir / "input_source_matmul_fallback_stitched.png",
        [
            ("input", image),
            ("source matmul", source_recon),
            ("fallback original", fallback_recon),
        ],
    )

    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["image"] = str(args.input_image)
    metrics["image_size"] = image_size
    wrappers = metrics.setdefault("wrappers", {})
    wrappers["source_matmul_attention"] = {
        "encoder_variant": "source_matmul_attention",
        "tokens_vs_titok_encode_reference": {
            "shape_reference": list(default_tokens.reshape(default_tokens.shape[0], -1).shape),
            "shape_candidate": list(source_tokens.reshape(source_tokens.shape[0], -1).shape),
            "exact_match": bool(torch.equal(default_tokens.reshape(default_tokens.shape[0], -1), source_tokens.reshape(source_tokens.shape[0], -1))),
            "exact_fraction": float(
                (
                    default_tokens.reshape(default_tokens.shape[0], -1)
                    == source_tokens.reshape(source_tokens.shape[0], -1)
                )
                .to(torch.float32)
                .mean()
                .item()
            ),
            "mismatch_count": int(
                (
                    default_tokens.reshape(default_tokens.shape[0], -1)
                    != source_tokens.reshape(source_tokens.shape[0], -1)
                )
                .sum()
                .item()
            ),
            "num_tokens": int(default_tokens.numel()),
        },
        "latent_vs_titok_encode_reference": compare_latent_tensors(reference_latent, source_latent),
    }
    wrappers["fallback_original_non_quantized"] = {
        "checkpoint": str(args.fallback_checkpoint),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(json.dumps({"output_dir": str(output_dir), "input_image": str(args.input_image)}, indent=2))


if __name__ == "__main__":
    main()
