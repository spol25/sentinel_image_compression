#!/usr/bin/env python3
"""Export fallback student encoder to INT8 TFLite using ai_edge_torch Option B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import ai_edge_torch
import tensorflow as tf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "fallback_ai_edge_pre_lowered_calib500"
DEFAULT_FALLBACK_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "_archive"
    / "2026-06-fallback-visual-evals"
    / "fallback_solution"
    / "1MSELoss_0.01perceptualloss_0.1gradloss_0.01ssimloss_1fourierloss_0.0075ganloss_SSIM_best.pt"
)
DEFAULT_CALIBRATION_MANIFEST = Path(
    "/Volumes/Media/snapshot_serengeti_balanced_ptq_dataset/manifests/calibration_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distill-repo-root", default="/Users/sruthipolali/Documents/Playground/sentinel-titok-distill", type=Path)
    parser.add_argument("--fallback-checkpoint", default=DEFAULT_FALLBACK_CHECKPOINT, type=Path)
    parser.add_argument("--calibration-manifest", default=DEFAULT_CALIBRATION_MANIFEST, type=Path)
    parser.add_argument("--calibration-count", default=500, type=int)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_image_nchw(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def load_calibration_images(path: Path, count: int) -> list[Path]:
    if path.is_dir():
        images = sorted(
            image_path
            for image_path in path.iterdir()
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    else:
        manifest = json.loads(path.read_text())
        images = [Path(image_path) for image_path in manifest["images"]]
    existing = [image_path for image_path in images if image_path.exists()]
    if len(existing) < count:
        raise RuntimeError(f"Need {count} calibration images from {path}, found {len(existing)}")
    return existing[:count]


def load_student_encoder(distill_repo_root: Path, checkpoint_path: Path) -> torch.nn.Module:
    prod_root = distill_repo_root / "TiTok-Distill-Prod" / "titok-distill-prod"
    one_d_tokenizer = prod_root / "1d-tokenizer"
    for path in (prod_root, one_d_tokenizer):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    from Encoders import GeminiV0Encoder

    encoder = GeminiV0Encoder(
        arch="101",
        pretrained=False,
        codebook_size=4096,
        latent_dim=128,
        freeze_backbone=False,
        dropout=0.1,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if any(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    student_state = {
        key.removeprefix("student_model."): value
        for key, value in checkpoint.items()
        if key.startswith("student_model.")
    }
    missing, unexpected = encoder.load_state_dict(student_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Fallback student checkpoint mismatch. Missing={missing}, unexpected={unexpected}")
    return encoder.eval().to("cpu")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = output_dir / "encoder_int8.tflite"
    metadata_path = output_dir / "metadata.json"
    if tflite_path.exists() and metadata_path.exists() and not args.force:
        print(f"[reuse] {tflite_path}")
        return

    calibration_images = load_calibration_images(args.calibration_manifest, args.calibration_count)
    encoder = load_student_encoder(args.distill_repo_root, args.fallback_checkpoint)
    sample_input = load_image_nchw(calibration_images[0], args.image_size)

    def representative_dataset():
        for image_path in calibration_images:
            yield [load_image_nchw(image_path, args.image_size).numpy().astype(np.float32)]

    tfl_converter_flags = {
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "representative_dataset": representative_dataset,
        "target_spec": {
            "supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
        },
        "inference_input_type": tf.int8,
        "inference_output_type": tf.int8,
    }

    with torch.no_grad():
        edge_model = ai_edge_torch.convert(
            encoder,
            (sample_input,),
            _ai_edge_converter_flags=tfl_converter_flags,
        )
    edge_model.export(str(tflite_path))

    metadata = {
        "status": "succeeded",
        "pipeline": "README_adv Stage 4 Option B through pre-lowered INT8 TFLite",
        "fallback_checkpoint": str(args.fallback_checkpoint),
        "calibration_source": str(args.calibration_manifest),
        "calibration_count": len(calibration_images),
        "image_size": args.image_size,
        "tflite_pre_lowered": str(tflite_path),
        "stopped_before": "Vela / Ethos-U lowering",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
