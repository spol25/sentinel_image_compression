#!/usr/bin/env python3
"""Compare a CM33/Ethos-U output capture against local TiTok flows."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from titok_deploy_tools.ptq_tools.ptq import (  # noqa: E402
    build_encoder_quantizer_split,
    calibrate_prepared_encoder,
    convert_encoder_after_ptq,
    export_encoder_program,
    load_manifest_records,
    prepare_exported_encoder_for_ptq,
)
from titok_deploy_tools.wrapper_tools.titok_env import add_titok_root_to_path  # noqa: E402
from titok_deploy_tools.wrapper_tools.utils import load_image  # noqa: E402


def tensor_stats(tensor: torch.Tensor) -> dict:
    value = tensor.detach().to("cpu", dtype=torch.float32)
    flat = value.reshape(-1)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "first_8": [float(x) for x in flat[:8].tolist()],
    }


def compare_pair(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref = reference.detach().to("cpu", dtype=torch.float32).reshape(-1)
    cand = candidate.detach().to("cpu", dtype=torch.float32).reshape(-1)
    if ref.numel() != cand.numel():
        return {
            "comparable": False,
            "reason": f"numel mismatch: {ref.numel()} vs {cand.numel()}",
        }
    diff = cand - ref
    abs_diff = torch.abs(diff)
    l2 = torch.linalg.vector_norm(diff).item()
    ref_norm = torch.linalg.vector_norm(ref).item()
    mse = torch.mean(diff * diff).item()
    return {
        "comparable": True,
        "max_abs_error": float(abs_diff.max().item()),
        "mean_abs_error": float(abs_diff.mean().item()),
        "l2_error": float(l2),
        "normalized_l2_error": float(l2 / max(ref_norm, 1e-12)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(ref.unsqueeze(0), cand.unsqueeze(0)).item()
        ),
    }


def normalize_runtime_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) == 1 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported ExecuTorch runtime output type: {type(output)}")


def try_run_pte(pte_path: Path, image: torch.Tensor) -> tuple[torch.Tensor | None, dict]:
    status: dict = {"attempted": True, "pte_path": str(pte_path)}
    if not pte_path.exists():
        status.update({"runnable": False, "error": f"PTE not found: {pte_path}"})
        return None, status
    try:
        from executorch.runtime import Runtime

        runtime = Runtime.get()
        program = runtime.load_program(str(pte_path))
        method = program.load_method("forward")
        output = normalize_runtime_output(method.execute((image,)))
        status.update({"runnable": True})
        return output.detach().to("cpu", dtype=torch.float32), status
    except Exception as exc:  # noqa: BLE001 - this is a capability probe.
        status.update(
            {
                "runnable": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return None, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--titok-root", required=True, type=Path)
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_s128_imagenet",
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        default=None,
        type=Path,
        help="Calibration manifest for rerunning PTQ locally when --pre-lowering-npz is not provided.",
    )
    parser.add_argument("--calibration-count", type=int, default=4)
    parser.add_argument("--board-npz", required=True, type=Path)
    parser.add_argument("--pte-path", default=None, type=Path)
    parser.add_argument(
        "--pre-lowering-npz",
        default=None,
        type=Path,
        help="Saved pre-partition output from lower_ethosu_titok_s128_encoder.py --capture-image.",
    )
    parser.add_argument(
        "--pre-lowering-key",
        default="output",
        help="Array key inside --pre-lowering-npz.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    board = torch.from_numpy(np.load(args.board_npz)["output"]).to(torch.float32)

    titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
    titok.requires_grad_(False)
    image_size = int(titok.config.dataset.preprocessing.crop_size)
    image = load_image(args.image, image_size).to("cpu")
    encoder_only, _, _ = build_encoder_quantizer_split(
        titok,
        encoder_variant="source_sdpa_attention",
    )
    encoder_only = encoder_only.eval().to("cpu")
    encoder_only.requires_grad_(False)

    with torch.no_grad():
        float_encoder_output = encoder_only(image).detach().to("cpu", dtype=torch.float32)

    compile_spec_flags = None
    quantized_output = None
    final_export_output = None
    saved_pre_lowering_output = None
    if args.pre_lowering_npz is not None:
        saved_pre_lowering_output = torch.from_numpy(
            np.load(args.pre_lowering_npz)[args.pre_lowering_key]
        ).to(torch.float32)
    else:
        if args.manifest is None:
            raise SystemExit("--manifest is required when --pre-lowering-npz is not provided.")
        calibration_images = load_manifest_records(args.manifest)[: args.calibration_count]
        exported_program = export_encoder_program(encoder_only, image)
        prepared_encoder, compile_spec = prepare_exported_encoder_for_ptq(
            exported_program,
            backend="ethosu",
            is_per_channel=True,
            quantization_profile="int8",
            ethos_target="ethos-u65-256",
            ethos_system_config="Ethos_U65_High_End",
            ethos_memory_mode="Dedicated_Sram_384KB",
            ethos_config_ini="Arm/vela.ini",
            ethos_extra_flags=[],
            quantize_matmul=False,
        )
        compile_spec_flags = compile_spec.compiler_flags if compile_spec is not None else None
        calibrate_prepared_encoder(prepared_encoder, calibration_images, image_size)
        quantized_encoder = convert_encoder_after_ptq(prepared_encoder, backend="ethosu")
        with torch.no_grad():
            quantized_output = quantized_encoder(image).detach().to("cpu", dtype=torch.float32)
        final_export = torch.export.export(quantized_encoder, (image,), strict=True)
        final_export_module = final_export.module()
        with torch.no_grad():
            final_export_output = final_export_module(image).detach().to("cpu", dtype=torch.float32)

    pte_output, pte_status = (
        try_run_pte(args.pte_path, image)
        if args.pte_path is not None
        else (None, {"attempted": False, "reason": "--pte-path was not provided"})
    )

    torch.save(float_encoder_output, output_dir / "serengeti_0345_float_encoder.pt")
    if saved_pre_lowering_output is not None:
        torch.save(saved_pre_lowering_output, output_dir / "serengeti_0345_saved_pre_lowering.pt")
        np.savez(output_dir / "serengeti_0345_saved_pre_lowering.npz", output=saved_pre_lowering_output.numpy())
    if quantized_output is not None:
        torch.save(quantized_output, output_dir / "serengeti_0345_quantized_pre_lowering.pt")
        np.savez(output_dir / "serengeti_0345_quantized_pre_lowering.npz", output=quantized_output.numpy())
    if final_export_output is not None:
        torch.save(final_export_output, output_dir / "serengeti_0345_final_export_pre_lowering.pt")
        np.savez(output_dir / "serengeti_0345_final_export_pre_lowering.npz", output=final_export_output.numpy())
    if pte_output is not None:
        torch.save(pte_output, output_dir / "serengeti_0345_local_pte_runtime.pt")
        np.savez(output_dir / "serengeti_0345_local_pte_runtime.npz", output=pte_output.numpy())

    report = {
        "image": str(args.image),
        "image_size": image_size,
        "board_npz": str(args.board_npz),
        "pte_path": str(args.pte_path) if args.pte_path is not None else None,
        "pre_lowering_npz": str(args.pre_lowering_npz) if args.pre_lowering_npz is not None else None,
        "compile_spec_flags": compile_spec_flags,
        "outputs": {
            "float_encoder": tensor_stats(float_encoder_output),
            "board_ethosu": tensor_stats(board),
        },
        "comparisons": {
            "board_vs_float_encoder": compare_pair(float_encoder_output, board),
        },
        "local_pte_runtime": pte_status,
    }
    if saved_pre_lowering_output is not None:
        report["outputs"]["saved_pre_lowering"] = tensor_stats(saved_pre_lowering_output)
        report["comparisons"]["board_vs_saved_pre_lowering"] = compare_pair(saved_pre_lowering_output, board)
        report["comparisons"]["saved_pre_lowering_vs_float_encoder"] = compare_pair(
            float_encoder_output,
            saved_pre_lowering_output,
        )
    if quantized_output is not None and final_export_output is not None:
        report["outputs"]["quantized_pre_lowering"] = tensor_stats(quantized_output)
        report["outputs"]["final_export_pre_lowering"] = tensor_stats(final_export_output)
        report["comparisons"]["board_vs_quantized_pre_lowering"] = compare_pair(quantized_output, board)
        report["comparisons"]["board_vs_final_export_pre_lowering"] = compare_pair(final_export_output, board)
        report["comparisons"]["quantized_pre_lowering_vs_float_encoder"] = compare_pair(
            float_encoder_output,
            quantized_output,
        )
        report["comparisons"]["final_export_pre_lowering_vs_quantized_pre_lowering"] = compare_pair(
            quantized_output,
            final_export_output,
        )
    if pte_output is not None:
        report["outputs"]["local_pte_runtime"] = tensor_stats(pte_output)
        report["comparisons"]["local_pte_runtime_vs_board"] = compare_pair(board, pte_output)
        report["comparisons"]["local_pte_runtime_vs_quantized_pre_lowering"] = compare_pair(
            quantized_output, pte_output
        )

    report_path = output_dir / "serengeti_0345_three_way_comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
