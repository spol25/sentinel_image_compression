#!/usr/bin/env python3
"""Export fallback student encoder to ONNX and INT8 TFLite before Vela lowering."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_fallback_vs_residual_add import (  # noqa: E402
    load_calibration_images,
    load_fallback_model,
)
from titok_deploy_tools.wrapper_tools.titok_env import add_titok_root_to_path  # noqa: E402
from titok_deploy_tools.wrapper_tools.utils import load_image  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "fallback_tflite_pre_lowered_calib500"
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
    parser.add_argument("--titok-root", default="/Users/sruthipolali/Documents/Playground/1d-tokenizer", type=Path)
    parser.add_argument("--repo-id", default="yucornetto/tokenizer_titok_s128_imagenet")
    parser.add_argument("--distill-repo-root", default="/Users/sruthipolali/Documents/Playground/sentinel-titok-distill", type=Path)
    parser.add_argument("--fallback-checkpoint", default=DEFAULT_FALLBACK_CHECKPOINT, type=Path)
    parser.add_argument("--calibration-manifest", default=DEFAULT_CALIBRATION_MANIFEST, type=Path)
    parser.add_argument("--calibration-count", default=500, type=int)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, log_path: Path | None = None) -> None:
    print("+", " ".join(cmd))
    if log_path is None:
        subprocess.run(cmd, check=True)
        return
    try:
        with log_path.open("w") as log_file:
            subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        if log_path.exists():
            lines = log_path.read_text(errors="replace").splitlines()
            print(f"[error] tail of {log_path}:")
            print("\n".join(lines[-120:]))
        raise


def venv_tool(name: str) -> str:
    for base in (Path(sys.prefix) / "bin", Path(sys.executable).parent, Path(sys.executable).resolve().parent):
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(name)


def representative_dataset(calibration_images: list[Path], image_size: int):
    for path in calibration_images:
        image = load_image(path, image_size)
        image = image.permute(0, 2, 3, 1).numpy().astype(np.float32)
        yield [image]


def patch_openvino_nchw_broadcasts(xml_path: Path, patched_xml_path: Path) -> int:
    """Patch NCHW channel-broadcast constants for OpenVINO2TF's NHWC graph build."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    patched = 0
    for layer in root.findall(".//layer"):
        for data in layer.findall("./data"):
            shape = data.attrib.get("shape")
            if not shape:
                continue
            dims = [int(item.strip()) for item in shape.split(",")]
            if len(dims) == 4 and dims[0] == 1 and dims[1] > 1 and dims[2] == 1 and dims[3] == 1:
                data.attrib["shape"] = f"1, 1, 1, {dims[1]}"
                patched += 1
        for port in layer.findall(".//port"):
            dims_el = port.findall("dim")
            dims = [int(dim.text) for dim in dims_el]
            if len(dims) == 4 and dims[0] == 1 and dims[1] > 1 and dims[2] == 1 and dims[3] == 1:
                dims_el[0].text = "1"
                dims_el[1].text = "1"
                dims_el[2].text = "1"
                dims_el[3].text = str(dims[1])
                patched += 1
    tree.write(patched_xml_path, encoding="UTF-8", xml_declaration=True)
    patched_bin_path = patched_xml_path.with_suffix(".bin")
    if patched_bin_path != xml_path.with_suffix(".bin"):
        shutil.copy2(xml_path.with_suffix(".bin"), patched_bin_path)
    return patched


def convert_saved_model_to_int8_tflite(
    *,
    saved_model_dir: Path,
    output_path: Path,
    calibration_images: list[Path],
    image_size: int,
) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(calibration_images, image_size)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def main() -> None:
    args = parse_args()
    add_titok_root_to_path(str(args.titok_root))
    from modeling.titok import TiTok

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "encoder.onnx"
    saved_model_dir = output_dir / "encoder"
    tflite_path = output_dir / "encoder_int8.tflite"
    metadata_path = output_dir / "metadata.json"
    openvino_xml = output_dir / "encoder.xml"
    patched_openvino_xml = output_dir / "encoder_nhwc_broadcast.xml"
    openvino2tf_log = output_dir / "openvino2tensorflow.log"

    if tflite_path.exists() and metadata_path.exists() and not args.force:
        print(f"[reuse] {tflite_path}")
        return

    titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
    titok.requires_grad_(False)
    image_size = int(titok.config.dataset.preprocessing.crop_size)
    fallback_model = load_fallback_model(args.distill_repo_root, titok, args.fallback_checkpoint)
    encoder = fallback_model.student_model.eval().to("cpu")
    encoder.requires_grad_(False)

    calibration_images, calibration_source = load_calibration_images(
        args.calibration_manifest, args.calibration_count
    )
    example_input = load_image(calibration_images[0], image_size).to("cpu")

    if not onnx_path.exists() or args.force:
        torch.onnx.export(
            encoder,
            example_input,
            onnx_path,
            export_params=True,
            input_names=["input"],
            output_names=["output"],
            opset_version=20,
        )

    saved_model_file = saved_model_dir / "saved_model.pb"
    if saved_model_dir.exists() and args.force:
        shutil.rmtree(saved_model_dir)
    if not saved_model_file.exists():
        if saved_model_dir.exists():
            shutil.rmtree(saved_model_dir)
        if not openvino_xml.exists() or args.force:
            run([venv_tool("ovc"), str(onnx_path), "--output_model", str(output_dir / "encoder")])
        if not patched_openvino_xml.exists() or args.force:
            patched_count = patch_openvino_nchw_broadcasts(openvino_xml, patched_openvino_xml)
            print(f"[patch] rewrote {patched_count} OpenVINO channel-broadcast dims for OpenVINO2TF")
        run(
            [
                venv_tool("openvino2tensorflow"),
                "--model_path",
                str(patched_openvino_xml),
                "--model_output_path",
                str(saved_model_dir),
                "--output_saved_model",
                "--non_verbose",
            ],
            log_path=openvino2tf_log,
        )

    convert_saved_model_to_int8_tflite(
        saved_model_dir=saved_model_dir,
        output_path=tflite_path,
        calibration_images=calibration_images,
        image_size=image_size,
    )

    metadata = {
        "status": "succeeded",
        "pipeline": "README_adv Stage 4 Option A through pre-lowered INT8 TFLite",
        "repo_id": args.repo_id,
        "fallback_checkpoint": str(args.fallback_checkpoint),
        "calibration_source": str(calibration_source),
        "calibration_count": len(calibration_images),
        "onnx": str(onnx_path),
        "openvino": str(openvino_xml),
        "openvino2tf_broadcast_patched": str(patched_openvino_xml),
        "openvino2tf_log": str(openvino2tf_log),
        "saved_model": str(saved_model_dir),
        "tflite_pre_lowered": str(tflite_path),
        "stopped_before": "Vela / Ethos-U lowering",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
