#!/usr/bin/env python3
"""Evaluate saved pre-lowered quantized TiTok/fallback artifacts on one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_fallback_vs_residual_add import (  # noqa: E402
    decode_titok_tokens,
    latent_to_tokens,
    load_fallback_model,
)
from titok_deploy_tools.ptq_tools.ptq import (  # noqa: E402
    build_encoder_quantizer_split,
    compare_latent_tensors,
)
from titok_deploy_tools.wrapper_tools.titok_env import add_titok_root_to_path  # noqa: E402
from titok_deploy_tools.wrapper_tools.utils import load_image, save_reconstruction  # noqa: E402


DEFAULT_COMPARISON_DIR = REPO_ROOT / "outputs" / "titok_vs_fallback_step_by_step_comparison"
DEFAULT_ARTIFACT_DIR = (
    REPO_ROOT
    / "outputs"
    / "_archive"
    / "2026-06-fallback-visual-evals"
    / "better_inputs_six_way_saved_ptq_calib500"
    / "calibrated_models"
)
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
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR, type=Path)
    parser.add_argument("--fallback-checkpoint", default=DEFAULT_FALLBACK_CHECKPOINT, type=Path)
    return parser.parse_args()


def load_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def safe_eval_cpu(module):
    try:
        module = module.eval()
    except NotImplementedError:
        pass
    try:
        module = module.to("cpu")
    except NotImplementedError:
        pass
    return module


def save_tokens(tokens: torch.Tensor, path: Path, metadata: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "shape": list(tokens.shape),
                "tokens": tokens.detach().to("cpu").tolist(),
                "metadata": metadata,
            },
            indent=2,
        )
        + "\n"
    )


def run_titok_encoder_artifact(
    *,
    artifact_dir: Path,
    key: str,
    image: torch.Tensor,
    titok,
    output_dir: Path,
    output_prefix: str,
    reference_latent: torch.Tensor,
) -> dict:
    artifact = artifact_dir / f"{key}.pt2"
    metadata = load_metadata(artifact_dir / f"{key}.json")
    module = safe_eval_cpu(torch.export.load(artifact).module())
    with torch.no_grad():
        latent = module(image).detach().to("cpu", dtype=torch.float32)
        tokens = latent_to_tokens(titok, latent)
        recon = decode_titok_tokens(titok, tokens)

    recon_path = output_dir / f"{output_prefix}_reconstruction.png"
    tokens_path = output_dir / f"{output_prefix}_tokens.json"
    save_reconstruction(recon, recon_path)
    save_tokens(tokens, tokens_path, metadata)
    return {
        "artifact": str(artifact),
        "metadata": metadata,
        "reconstruction": str(recon_path),
        "tokens": str(tokens_path),
        "latent_vs_float_source_matmul": compare_latent_tensors(reference_latent, latent),
    }


def update_metrics_json(output_dir: Path, full_a16w8_run: dict, residual_run: dict) -> None:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    else:
        metrics = {}
    wrappers = metrics.setdefault("wrappers", {})
    wrappers.setdefault("quantized_full_a16w8", {})[
        "latent_vs_titok_encode_reference"
    ] = full_a16w8_run["latent_vs_float_source_matmul"]
    wrappers.setdefault("quantized_residual_add_a16w8", {})[
        "latent_vs_titok_encode_reference"
    ] = residual_run["latent_vs_float_source_matmul"]
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")


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
    reference_encoder, _, _ = build_encoder_quantizer_split(titok, encoder_variant="baseline")
    reference_encoder = reference_encoder.eval().to("cpu")
    reference_encoder.requires_grad_(False)
    with torch.no_grad():
        reference_latent = reference_encoder(image).detach().to("cpu", dtype=torch.float32)

    full_a16w8_run = run_titok_encoder_artifact(
        artifact_dir=args.artifact_dir,
        key="bhld_matmul_attention_a16w8",
        image=image,
        titok=titok,
        output_dir=output_dir,
        output_prefix="quantized_full_a16w8",
        reference_latent=reference_latent,
    )
    residual_run = run_titok_encoder_artifact(
        artifact_dir=args.artifact_dir,
        key="bhld_matmul_residual_add_a16w8",
        image=image,
        titok=titok,
        output_dir=output_dir,
        output_prefix="quantized_residual_add_a16w8",
        reference_latent=reference_latent,
    )
    update_metrics_json(output_dir, full_a16w8_run, residual_run)

    fallback_artifact = args.artifact_dir / "fallback_quantized_student_qnnpack_torchscript.pt"
    fallback_metadata = load_metadata(args.artifact_dir / "fallback_quantized_student_qnnpack.json")
    torch.backends.quantized.engine = "qnnpack"
    fallback_model = load_fallback_model(args.distill_repo_root, titok, args.fallback_checkpoint)
    fallback_model.student_model = torch.jit.load(fallback_artifact, map_location="cpu").eval()
    fallback_model = safe_eval_cpu(fallback_model)
    with torch.no_grad():
        fallback_recon, fallback_tokens = fallback_model(image)

    fallback_recon_path = output_dir / "quantized_fallback_qnnpack_reconstruction.png"
    fallback_tokens_path = output_dir / "quantized_fallback_qnnpack_tokens.json"
    save_reconstruction(fallback_recon, fallback_recon_path)
    save_tokens(fallback_tokens.to("cpu"), fallback_tokens_path, fallback_metadata)

    run_metadata = {
        "input_image": str(args.input_image),
        "repo_id": args.repo_id,
        "artifact_dir": str(args.artifact_dir),
        "fallback_artifact": str(fallback_artifact),
        "full_a16w8_artifact": full_a16w8_run["artifact"],
        "residual_artifact": residual_run["artifact"],
        "full_a16w8_metadata": full_a16w8_run["metadata"],
        "residual_metadata": residual_run["metadata"],
        "fallback_metadata": fallback_metadata,
        "outputs": {
            "full_a16w8_reconstruction": full_a16w8_run["reconstruction"],
            "full_a16w8_tokens": full_a16w8_run["tokens"],
            "residual_reconstruction": residual_run["reconstruction"],
            "residual_tokens": residual_run["tokens"],
            "fallback_reconstruction": str(fallback_recon_path),
            "fallback_tokens": str(fallback_tokens_path),
        },
    }
    (output_dir / "quantized_pre_lowered_run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    print(json.dumps(run_metadata["outputs"], indent=2))


if __name__ == "__main__":
    main()
