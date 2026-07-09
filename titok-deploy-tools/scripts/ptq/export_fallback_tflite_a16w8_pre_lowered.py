#!/usr/bin/env python3
"""Convert the fallback SavedModel to A16W8 INT16-activation/INT8-weight TFLite."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import tensorflow as tf  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_fallback_vs_residual_add import load_calibration_images  # noqa: E402
from titok_deploy_tools.wrapper_tools.utils import load_image  # noqa: E402


DEFAULT_SOURCE_DIR = REPO_ROOT / "outputs" / "fallback_tflite_pre_lowered_calib500"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "fallback_tflite_a16w8_pre_lowered_calib500"
DEFAULT_CALIBRATION_MANIFEST = Path(
    "/Volumes/Media/snapshot_serengeti_balanced_ptq_dataset/manifests/calibration_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR, type=Path)
    parser.add_argument("--calibration-manifest", default=DEFAULT_CALIBRATION_MANIFEST, type=Path)
    parser.add_argument("--calibration-count", default=500, type=int)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def representative_dataset(calibration_images: list[Path], image_size: int):
    for image_path in calibration_images:
        image = load_image(image_path, image_size)
        image = image.permute(0, 2, 3, 1).numpy().astype(np.float32)
        yield [image]


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir if args.source_dir.is_absolute() else REPO_ROOT / args.source_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    saved_model_dir = source_dir / "encoder"
    source_metadata_path = source_dir / "metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = output_dir / "encoder_a16w8.tflite"
    metadata_path = output_dir / "metadata.json"

    if tflite_path.exists() and metadata_path.exists() and not args.force:
        print(f"[reuse] {tflite_path}")
        return
    if not (saved_model_dir / "saved_model.pb").exists():
        raise FileNotFoundError(saved_model_dir / "saved_model.pb")

    calibration_images, calibration_source = load_calibration_images(
        args.calibration_manifest, args.calibration_count
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(calibration_images, args.image_size)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]
    converter.inference_input_type = tf.int16
    converter.inference_output_type = tf.int16
    tflite_path.write_bytes(converter.convert())

    source_metadata = json.loads(source_metadata_path.read_text()) if source_metadata_path.exists() else {}
    metadata = {
        "status": "succeeded",
        "pipeline": "Fallback SavedModel to pre-lowered A16W8 TFLite",
        "quantization": "A16W8 / int16 activations, int8 weights",
        "source_metadata": source_metadata,
        "saved_model": str(saved_model_dir),
        "calibration_source": str(calibration_source),
        "calibration_count": len(calibration_images),
        "image_size": args.image_size,
        "tflite_pre_lowered": str(tflite_path),
        "stopped_before": "Vela / Ethos-U lowering",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
