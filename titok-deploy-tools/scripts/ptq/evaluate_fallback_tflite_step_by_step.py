#!/usr/bin/env python3
"""Evaluate the README-style pre-lowered fallback INT8 TFLite encoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import tensorflow as tf  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_fallback_vs_residual_add import decode_titok_tokens  # noqa: E402
from titok_deploy_tools.wrapper_tools.titok_env import add_titok_root_to_path  # noqa: E402
from titok_deploy_tools.wrapper_tools.utils import load_image, save_reconstruction  # noqa: E402


DEFAULT_COMPARISON_DIR = REPO_ROOT / "outputs" / "titok_vs_fallback_step_by_step_comparison"
DEFAULT_INPUT = REPO_ROOT / "outputs" / "better_inputs" / "0026_day_near_S2_H08_R3_IMAG0666.jpg"
DEFAULT_TFLITE = REPO_ROOT / "outputs" / "fallback_tflite_pre_lowered_calib500" / "encoder_int8.tflite"
DEFAULT_TFLITE_METADATA = REPO_ROOT / "outputs" / "fallback_tflite_pre_lowered_calib500" / "metadata.json"
DEFAULT_OUTPUT_PREFIX = "quantized_fallback_tflite_pre_lowered"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--titok-root", default="/Users/sruthipolali/Documents/Playground/1d-tokenizer", type=Path)
    parser.add_argument("--repo-id", default="yucornetto/tokenizer_titok_s128_imagenet")
    parser.add_argument("--input-image", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_COMPARISON_DIR, type=Path)
    parser.add_argument("--tflite-model", default=DEFAULT_TFLITE, type=Path)
    parser.add_argument("--tflite-metadata", default=DEFAULT_TFLITE_METADATA, type=Path)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def quantize_for_input(image: torch.Tensor, input_detail: dict) -> np.ndarray:
    scale, zero_point = input_detail["quantization"]
    if scale <= 0:
        raise ValueError(f"Expected quantized TFLite input, got quantization={input_detail['quantization']}")
    input_shape = list(input_detail["shape"])
    if input_shape == list(image.shape):
        array = image.detach().cpu().numpy().astype(np.float32)
    elif input_shape == [image.shape[0], image.shape[2], image.shape[3], image.shape[1]]:
        array = image.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)
    else:
        raise ValueError(f"Unsupported TFLite input shape {input_shape} for image shape {list(image.shape)}")
    quantized = np.round(array / scale + zero_point)
    dtype = input_detail["dtype"]
    info = np.iinfo(dtype)
    return np.clip(quantized, info.min, info.max).astype(dtype)


def dequantize_output(output: np.ndarray, output_detail: dict) -> torch.Tensor:
    scale, zero_point = output_detail["quantization"]
    if scale <= 0:
        return torch.from_numpy(output.astype(np.float32))
    dequantized = (output.astype(np.float32) - zero_point) * scale
    return torch.from_numpy(dequantized)


def save_tokens(tokens: torch.Tensor, path: Path, metadata: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "shape": list(tokens.shape),
                "tokens": tokens.detach().cpu().tolist(),
                "metadata": metadata,
            },
            indent=2,
        )
        + "\n"
    )


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

    interpreter = tf.lite.Interpreter(
        model_path=str(args.tflite_model),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_detail["index"], quantize_for_input(image, input_detail))
    interpreter.invoke()

    logits = dequantize_output(interpreter.get_tensor(output_detail["index"]), output_detail)
    tokens = torch.argmax(logits, dim=1).to(torch.long)
    recon = decode_titok_tokens(titok, tokens)

    metadata = json.loads(args.tflite_metadata.read_text()) if args.tflite_metadata.exists() else {}
    metadata.update(
        {
            "artifact": str(args.tflite_model),
            "input_detail": {
                "name": input_detail["name"],
                "shape": input_detail["shape"].tolist(),
                "dtype": str(input_detail["dtype"]),
                "quantization": list(input_detail["quantization"]),
            },
            "output_detail": {
                "name": output_detail["name"],
                "shape": output_detail["shape"].tolist(),
                "dtype": str(output_detail["dtype"]),
                "quantization": list(output_detail["quantization"]),
            },
        }
    )

    recon_path = output_dir / f"{args.output_prefix}_reconstruction.png"
    tokens_path = output_dir / f"{args.output_prefix}_tokens.json"
    run_path = output_dir / f"{args.output_prefix}_run.json"
    save_reconstruction(recon, recon_path)
    save_tokens(tokens, tokens_path, metadata)

    run_metadata = {
        "input_image": str(args.input_image),
        "repo_id": args.repo_id,
        "tflite_model": str(args.tflite_model),
        "tflite_metadata": str(args.tflite_metadata),
        "reconstruction": str(recon_path),
        "tokens": str(tokens_path),
        "input_detail": metadata["input_detail"],
        "output_detail": metadata["output_detail"],
    }
    run_path.write_text(json.dumps(run_metadata, indent=2) + "\n")
    print(json.dumps(run_metadata, indent=2))


if __name__ == "__main__":
    main()
