#!/usr/bin/env python3
"""Calibrate/save six TiTok encoder variants and compare reconstructions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_fallback_vs_residual_add import (  # noqa: E402
    DistillEncTiTokDec,
    decode_titok_tokens,
    latent_to_tokens,
    load_calibration_images,
    load_fallback_model,
    make_panel,
    quantize_fallback_student,
    reconstruction_metrics,
)
from titok_deploy_tools.ptq_tools.ptq import (  # noqa: E402
    build_encoder_quantizer_split,
    calibrate_prepared_encoder,
    convert_encoder_after_ptq,
    export_encoder_program,
    prepare_exported_encoder_for_ptq,
)
from titok_deploy_tools.wrapper_tools.titok_env import add_titok_root_to_path  # noqa: E402
from titok_deploy_tools.wrapper_tools.utils import (  # noqa: E402
    load_image,
    resolve_input_path,
    resolve_output_dir,
    save_reconstruction,
)


@dataclass(frozen=True)
class TitokVariantSpec:
    key: str
    label: str
    encoder_variant: str
    quantization_profile: str | None


TITOK_VARIANTS = (
    TitokVariantSpec(
        key="original_attention_a16w8",
        label="original attention A16W8",
        encoder_variant="baseline",
        quantization_profile="a16w8",
    ),
    TitokVariantSpec(
        key="einsum_attention_a16w8",
        label="einsum attention A16W8",
        encoder_variant="einsum_attention",
        quantization_profile="a16w8",
    ),
    TitokVariantSpec(
        key="bhld_matmul_attention_a16w8",
        label="BHLD matmul attention A16W8",
        encoder_variant="source_matmul_attention",
        quantization_profile="a16w8",
    ),
    TitokVariantSpec(
        key="bhld_matmul_residual_add_a16w8",
        label="BHLD matmul residual-add A16W8",
        encoder_variant="source_matmul_attention",
        quantization_profile="int8_surface_transformer_norm_residual_a16w8",
    ),
    TitokVariantSpec(
        key="float_original_attention",
        label="float original attention",
        encoder_variant="baseline",
        quantization_profile=None,
    ),
)


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
        default=REPO_ROOT / "outputs/better_inputs_six_way_saved_ptq_calib500",
        type=Path,
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        type=Path,
        help="Directory containing saved calibrated artifacts. Defaults to output-dir/calibrated_models.",
    )
    parser.add_argument("--calibration-count", default=500, type=int)
    parser.add_argument(
        "--force-recalibrate",
        action="store_true",
        help="Ignore saved calibrated artifacts and rebuild them.",
    )
    return parser.parse_args()


def load_torch_artifact(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def safe_eval_cpu(module: nn.Module) -> nn.Module:
    try:
        module = module.eval()
    except NotImplementedError:
        pass
    try:
        module = module.to("cpu")
    except NotImplementedError:
        pass
    return module


def save_torch_artifact(module: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(safe_eval_cpu(module), path)


def load_exported_encoder(path: Path) -> nn.Module:
    return safe_eval_cpu(torch.export.load(path).module())


def save_exported_encoder(module: nn.Module, example_input: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exported_program = torch.export.export(module, (example_input,), strict=True)
    torch.export.save(exported_program, path)


def build_quantized_titok_encoder_exported_program(
    encoder_only: nn.Module,
    example_input: torch.Tensor,
    calibration_images: list[Path],
    image_size: int,
    quantization_profile: str,
):
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
    return torch.export.export(quantized_encoder, (example_input,), strict=True)


def artifact_metadata(
    *,
    kind: str,
    label: str,
    calibration_source: Path,
    calibration_count: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": label,
        "calibration_source": str(calibration_source),
        "calibration_count": calibration_count,
        **extra,
    }


def get_or_build_titok_variant(
    *,
    spec: TitokVariantSpec,
    titok,
    example_input: torch.Tensor,
    calibration_images: list[Path],
    calibration_source: Path,
    image_size: int,
    artifact_dir: Path,
    force_recalibrate: bool,
) -> nn.Module:
    artifact_path = artifact_dir / f"{spec.key}.pt2"
    metadata_path = artifact_dir / f"{spec.key}.json"
    if artifact_path.exists() and not force_recalibrate:
        print(f"[reuse] {spec.label}: {artifact_path}")
        return load_exported_encoder(artifact_path)

    encoder_only, _, _ = build_encoder_quantizer_split(
        titok,
        encoder_variant=spec.encoder_variant,
    )
    encoder_only = encoder_only.eval().to("cpu")
    encoder_only.requires_grad_(False)

    if spec.quantization_profile is None:
        print(f"[save] {spec.label}: float artifact")
        save_exported_encoder(encoder_only, example_input, artifact_path)
    else:
        print(f"[calibrate] {spec.label}: profile={spec.quantization_profile}")
        exported_program = build_quantized_titok_encoder_exported_program(
            encoder_only,
            example_input,
            calibration_images,
            image_size,
            spec.quantization_profile,
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.export.save(exported_program, artifact_path)

    metadata_path.write_text(
        json.dumps(
            artifact_metadata(
                kind="titok_encoder",
                label=spec.label,
                calibration_source=calibration_source,
                calibration_count=len(calibration_images) if spec.quantization_profile is not None else 0,
                extra={
                    "key": spec.key,
                    "encoder_variant": spec.encoder_variant,
                    "quantization_profile": spec.quantization_profile,
                    "artifact": str(artifact_path),
                },
            ),
            indent=2,
        )
        + "\n"
    )
    return load_exported_encoder(artifact_path)


def get_or_build_fallback(
    *,
    titok,
    distill_repo_root: Path,
    checkpoint_path: Path,
    calibration_images: list[Path],
    calibration_source: Path,
    image_size: int,
    artifact_dir: Path,
    force_recalibrate: bool,
) -> DistillEncTiTokDec:
    artifact_path = artifact_dir / "fallback_quantized_student_qnnpack_torchscript.pt"
    metadata_path = artifact_dir / "fallback_quantized_student_qnnpack.json"
    if artifact_path.exists() and not force_recalibrate:
        print(f"[reuse] fallback quantization: {artifact_path}")
        torch.backends.quantized.engine = "qnnpack"
        fallback_model = load_fallback_model(distill_repo_root, titok, checkpoint_path)
        fallback_model.student_model = torch.jit.load(artifact_path, map_location="cpu").eval()
        return safe_eval_cpu(fallback_model)

    print("[calibrate] fallback student: FX static INT8 qnnpack")
    fallback_model = load_fallback_model(distill_repo_root, titok, checkpoint_path)
    fallback_model = quantize_fallback_student(fallback_model, calibration_images, image_size)
    example_input = load_image(calibration_images[0], image_size).to("cpu")
    traced_student = torch.jit.trace(fallback_model.student_model.eval().to("cpu"), example_input)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    traced_student.save(str(artifact_path))
    metadata_path.write_text(
        json.dumps(
            artifact_metadata(
                kind="fallback_distill_model",
                label="fallback quantized student",
                calibration_source=calibration_source,
                calibration_count=len(calibration_images),
                extra={
                    "key": "fallback_quantized_student",
                    "checkpoint": str(checkpoint_path),
                    "quantization": "torch_fx_static_int8_qnnpack_student_encoder",
                    "artifact": str(artifact_path),
                },
            ),
            indent=2,
        )
        + "\n"
    )
    return safe_eval_cpu(fallback_model)


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    variants = sorted({str(row["variant"]) for row in rows})
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            metric: float(np.mean([float(row[metric]) for row in variant_rows]))
            for metric in ("psnr", "ssim", "mae", "rmse")
        }
    return summary


def main() -> None:
    args = parse_args()
    add_titok_root_to_path(str(args.titok_root))
    from modeling.titok import TiTok

    output_dir = resolve_output_dir(REPO_ROOT, str(args.output_dir))
    artifact_dir = args.artifact_dir or (output_dir / "calibrated_models")
    image_paths = sorted(
        path for path in args.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    print(f"[setup] Loading TiTok from {args.repo_id}")
    titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
    titok.requires_grad_(False)
    image_size = int(titok.config.dataset.preprocessing.crop_size)

    calibration_path = resolve_input_path(str(args.calibration_manifest), REPO_ROOT)
    calibration_images, calibration_source = load_calibration_images(calibration_path, args.calibration_count)
    example_input = load_image(calibration_images[0], image_size).to("cpu")

    titok_modules = {
        spec.key: get_or_build_titok_variant(
            spec=spec,
            titok=titok,
            example_input=example_input,
            calibration_images=calibration_images,
            calibration_source=calibration_source,
            image_size=image_size,
            artifact_dir=artifact_dir,
            force_recalibrate=args.force_recalibrate,
        )
        for spec in TITOK_VARIANTS
    }
    fallback_model = get_or_build_fallback(
        titok=titok,
        distill_repo_root=args.distill_repo_root,
        checkpoint_path=args.fallback_checkpoint,
        calibration_images=calibration_images,
        calibration_source=calibration_source,
        image_size=image_size,
        artifact_dir=artifact_dir,
        force_recalibrate=args.force_recalibrate,
    )

    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths):
        image = load_image(image_path, image_size).to("cpu")
        stem = f"{index:03d}_{image_path.stem}"
        reference_path = output_dir / "reconstructions" / stem / "reference.png"
        save_reconstruction(image, reference_path)
        panel_items = [("reference", image)]

        for spec in TITOK_VARIANTS:
            module = titok_modules[spec.key]
            with torch.no_grad():
                latent = module(image).detach().to("cpu", dtype=torch.float32)
                tokens = latent_to_tokens(titok, latent)
                recon = decode_titok_tokens(titok, tokens)
            recon_path = output_dir / "reconstructions" / stem / f"{spec.key}.png"
            save_reconstruction(recon, recon_path)
            metrics = reconstruction_metrics(image, recon)
            panel_items.append((spec.label, recon))
            rows.append(
                {
                    "index": index,
                    "image": str(image_path),
                    "stem": stem,
                    "variant": spec.key,
                    "label": spec.label,
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "reference_png": str(reference_path),
                    "reconstruction_png": str(recon_path),
                }
            )

        with torch.no_grad():
            fallback_recon, _ = fallback_model(image)
        fallback_path = output_dir / "reconstructions" / stem / "fallback_quantized_student.png"
        save_reconstruction(fallback_recon, fallback_path)
        fallback_metrics = reconstruction_metrics(image, fallback_recon)
        panel_items.append(("fallback quantized", fallback_recon))
        rows.append(
            {
                "index": index,
                "image": str(image_path),
                "stem": stem,
                "variant": "fallback_quantized_student",
                "label": "fallback quantized student",
                "psnr": fallback_metrics["psnr"],
                "ssim": fallback_metrics["ssim"],
                "mae": fallback_metrics["mae"],
                "rmse": fallback_metrics["rmse"],
                "reference_png": str(reference_path),
                "reconstruction_png": str(fallback_path),
            }
        )
        make_panel(output_dir / "panels" / f"{stem}_six_way.png", panel_items)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "succeeded",
        "image_dir": str(args.image_dir),
        "image_count": len(image_paths),
        "calibration_manifest": str(calibration_source),
        "calibration_count": len(calibration_images),
        "artifact_dir": str(artifact_dir),
        "fallback_checkpoint": str(args.fallback_checkpoint),
        "variants": [
            {
                "key": spec.key,
                "label": spec.label,
                "encoder_variant": spec.encoder_variant,
                "quantization_profile": spec.quantization_profile,
            }
            for spec in TITOK_VARIANTS
        ]
        + [
            {
                "key": "fallback_quantized_student",
                "label": "fallback quantized student",
                "quantization": "torch_fx_static_int8_qnnpack_student_encoder",
            }
        ],
        "mean": summarize(rows),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["mean"], indent=2))
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote reconstructions under {output_dir / 'reconstructions'}")


if __name__ == "__main__":
    main()
