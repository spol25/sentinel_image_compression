#!/usr/bin/env python3
"""Controlled host capture and lowering for board comparison.

This script intentionally keeps the host PTQ capture and lowered PTE creation in
one invocation so their provenance is auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

for parent in SCRIPT_DIR.parents:
    executorch_src = parent / "executorch-main" / "src"
    if executorch_src.exists():
        executorch_src_str = str(executorch_src)
        if executorch_src_str in sys.path:
            sys.path.remove(executorch_src_str)
        sys.path.insert(0, executorch_src_str)
        break

import numpy as np
import torch
from executorch.backends.arm.ethosu import EthosUPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.fx import Interpreter

from titok_deploy_tools.lowering_tools.ethosu_compat import EthosUCompatCompileSpec
from titok_deploy_tools.lowering_tools.executorch_summary import summarize_executorch_program
from titok_deploy_tools.lowering_tools.graph_summary import summarize_fx_graph
from titok_deploy_tools.lowering_tools.post_partition_qdq_fix import ReplaceSurvivingQdqWithOutVarPass
from titok_deploy_tools.ptq_tools.ptq import (
    build_encoder_quantizer_split,
    calibrate_prepared_encoder,
    convert_encoder_after_ptq,
    export_encoder_program,
    load_manifest_records,
    prepare_exported_encoder_for_ptq,
)
from titok_deploy_tools.wrapper_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.wrapper_tools.utils import load_image, resolve_input_path, resolve_output_dir


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().to("cpu")
    flat = value.reshape(-1)
    stats: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
    }
    if value.numel():
        if value.is_floating_point():
            numeric = flat.to(torch.float32)
            stats.update(
                {
                    "min": float(numeric.min().item()),
                    "max": float(numeric.max().item()),
                    "mean": float(numeric.mean().item()),
                    "std": float(numeric.std(unbiased=False).item()),
                    "first_16": [float(x) for x in numeric[:16].tolist()],
                }
            )
        else:
            numeric = flat.to(torch.int32)
            stats.update(
                {
                    "min": int(numeric.min().item()),
                    "max": int(numeric.max().item()),
                    "mean": float(numeric.to(torch.float32).mean().item()),
                    "first_16": [int(x) for x in numeric[:16].tolist()],
                }
            )
    return stats


def save_float_npz(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, output=tensor.detach().to("cpu", dtype=torch.float32).contiguous().numpy())


def save_int_npz(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().to("cpu").contiguous()
    if value.dtype == torch.int8:
        array = value.numpy()
    elif value.dtype == torch.uint8:
        array = value.numpy()
    else:
        array = value.to(torch.int16).numpy()
    np.savez(path, output_int=array)


def _shape_from_node(node) -> str:  # noqa: ANN001
    value = node.meta.get("val") if hasattr(node, "meta") else None
    if isinstance(value, torch.Tensor):
        return "x".join(str(dim) for dim in value.shape)
    return ""


def _node_arg_value(arg, module: torch.fx.GraphModule):  # noqa: ANN001
    if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
        target = arg.target
        value = module
        for part in str(target).split("."):
            value = getattr(value, part)
        return value
    return arg


def _qparam_scalar_or_stats(value) -> dict[str, str]:  # noqa: ANN001
    if isinstance(value, torch.Tensor):
        detached = value.detach().to("cpu")
        flat = detached.reshape(-1)
        result = {
            "value": "",
            "shape": "x".join(str(dim) for dim in detached.shape),
            "min": "",
            "max": "",
            "first_8": "",
        }
        if detached.numel():
            if detached.is_floating_point():
                numeric = flat.to(torch.float32)
                result.update(
                    {
                        "min": str(float(numeric.min().item())),
                        "max": str(float(numeric.max().item())),
                        "first_8": json.dumps([float(x) for x in numeric[:8].tolist()]),
                    }
                )
            else:
                numeric = flat.to(torch.int64)
                result.update(
                    {
                        "min": str(int(numeric.min().item())),
                        "max": str(int(numeric.max().item())),
                        "first_8": json.dumps([int(x) for x in numeric[:8].tolist()]),
                    }
                )
        return result
    return {
        "value": str(value),
        "shape": "",
        "min": "",
        "max": "",
        "first_8": "",
    }


def dump_qparams_csv(path: Path, module: torch.fx.GraphModule, stage: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for node in module.graph.nodes:
        target = str(node.target)
        if "quantize_per_tensor" not in target and "dequantize_per_tensor" not in target:
            if "quantize_per_channel" not in target and "dequantize_per_channel" not in target:
                continue

        args = list(node.args)
        op_kind = "dequantize" if "dequantize" in target else "quantize"
        qscheme = "per_channel" if "per_channel" in target else "per_tensor"
        scale = zp = axis = qmin = qmax = dtype = ""

        if qscheme == "per_tensor" and len(args) >= 6:
            scale = _node_arg_value(args[1], module)
            zp = _node_arg_value(args[2], module)
            qmin = _node_arg_value(args[3], module)
            qmax = _node_arg_value(args[4], module)
            dtype = args[5]
        elif qscheme == "per_channel" and len(args) >= 7:
            scale = _node_arg_value(args[1], module)
            zp = _node_arg_value(args[2], module)
            axis = _node_arg_value(args[3], module)
            qmin = _node_arg_value(args[4], module)
            qmax = _node_arg_value(args[5], module)
            dtype = args[6]

        scale_stats = _qparam_scalar_or_stats(scale)
        zp_stats = _qparam_scalar_or_stats(zp)
        rows.append(
            {
                "stage": stage,
                "node_name": node.name,
                "op_kind": op_kind,
                "qscheme": qscheme,
                "target": target,
                "tensor_shape": _shape_from_node(node),
                "scale": scale_stats["value"],
                "scale_shape": scale_stats["shape"],
                "scale_min": scale_stats["min"],
                "scale_max": scale_stats["max"],
                "scale_first_8": scale_stats["first_8"],
                "zero_point": zp_stats["value"],
                "zero_point_shape": zp_stats["shape"],
                "zero_point_min": zp_stats["min"],
                "zero_point_max": zp_stats["max"],
                "zero_point_first_8": zp_stats["first_8"],
                "axis": str(axis),
                "qmin": str(qmin),
                "qmax": str(qmax),
                "dtype": str(dtype),
                "args": json.dumps([str(arg) for arg in args]),
            }
        )

    fieldnames = [
        "stage",
        "node_name",
        "op_kind",
        "qscheme",
        "target",
        "tensor_shape",
        "scale",
        "scale_shape",
        "scale_min",
        "scale_max",
        "scale_first_8",
        "zero_point",
        "zero_point_shape",
        "zero_point_min",
        "zero_point_max",
        "zero_point_first_8",
        "axis",
        "qmin",
        "qmax",
        "dtype",
        "args",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


class QuantizeCaptureInterpreter(Interpreter):
    def __init__(self, module: torch.fx.GraphModule):
        super().__init__(module)
        self.quantize_records: list[dict[str, Any]] = []

    def run_node(self, node):  # noqa: ANN001
        result = super().run_node(node)
        target = str(node.target)
        if (
            "quantize_per_tensor" in target
            and "dequantize_per_tensor" not in target
            and isinstance(result, torch.Tensor)
        ):
            self.quantize_records.append(
                {
                    "name": node.name,
                    "target": target,
                    "shape": list(result.shape),
                    "dtype": str(result.dtype),
                    "value": result.detach().to("cpu").clone(),
                    "args": [str(arg) for arg in node.args],
                }
            )
        return result


def capture_last_matching_quantize(module: torch.fx.GraphModule, image: torch.Tensor, output_shape: list[int]):
    interp = QuantizeCaptureInterpreter(module)
    _ = interp.run(image)
    matching = [
        record
        for record in interp.quantize_records
        if record["shape"] == output_shape and isinstance(record["value"], torch.Tensor)
    ]
    if not matching:
        raise RuntimeError(f"No quantize_per_tensor node found with output shape {output_shape}")
    return matching[-1], interp.quantize_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--titok-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="yucornetto/tokenizer_titok_s128_imagenet")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--calibration-count", type=int, default=4)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--encoder-variant",
        default="source_sdpa_attention",
        choices=(
            "baseline",
            "reshape_batch",
            "bmm_attention",
            "source_matmul_attention",
            "source_query_chunked_matmul_attention",
            "source_sdpa_attention",
            "einsum_attention",
        ),
    )
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--prefix-num-blocks", type=int)
    parser.add_argument("--artifact-name", default="controlled_sdpa_int8_dedicated_sram_384kb.pte")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_input_path(str(args.manifest), REPO_ROOT)
    image_path = resolve_input_path(str(args.image), REPO_ROOT)
    calibration_images = load_manifest_records(manifest_path)[: args.calibration_count]
    if not calibration_images:
        raise SystemExit("Calibration image list is empty.")

    report: dict[str, Any] = {
        "status": "started",
        "repo_id": args.repo_id,
        "encoder_variant": args.encoder_variant,
        "query_chunk_size": args.query_chunk_size,
        "prefix_num_blocks": args.prefix_num_blocks,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "calibration_count": len(calibration_images),
        "calibration_images": [str(p) for p in calibration_images],
        "image": str(image_path),
        "quantizer_backend": "ethosu",
        "quantization_profile": "int8",
        "per_channel": True,
        "quantize_matmul": False,
        "ethos_target": "ethos-u65-256",
        "ethos_system_config": "Ethos_U65_High_End",
        "ethos_memory_mode": "Dedicated_Sram_384KB",
        "ethos_config_ini": "Arm/vela.ini",
    }
    report_path = output_dir / "controlled_host_capture_and_lower_summary.json"

    try:
        print("[1/9] Loading TiTok model")
        titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
        titok.requires_grad_(False)
        image_size = int(titok.config.dataset.preprocessing.crop_size)
        report["image_size"] = image_size

        print(f"[2/9] Building {args.encoder_variant} encoder")
        encoder_only, _, _ = build_encoder_quantizer_split(
            titok,
            encoder_variant=args.encoder_variant,
            prefix_num_blocks=args.prefix_num_blocks,
            query_chunk_size=args.query_chunk_size,
        )
        encoder_only = encoder_only.eval().to("cpu")
        encoder_only.requires_grad_(False)
        report["wrapper_variant"] = encoder_only.__class__.__name__

        image = load_image(image_path, image_size).to("cpu")
        example_input = load_image(calibration_images[0], image_size).to("cpu")

        print("[3/9] Running float encoder")
        with torch.no_grad():
            float_output = encoder_only(image).detach().to("cpu", dtype=torch.float32)
        float_path = output_dir / "host_float_encoder_output.npz"
        save_float_npz(float_path, float_output)
        report["host_float_encoder_output"] = {
            "path": str(float_path),
            "stats": tensor_stats(float_output),
        }

        print("[4/9] Exporting and preparing PTQ")
        exported_program = export_encoder_program(encoder_only, example_input)
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
        report["compile_spec_flags"] = compile_spec.compiler_flags if compile_spec is not None else None

        print(f"[5/9] Calibrating on {len(calibration_images)} fixed image(s)")
        calibrate_prepared_encoder(prepared_encoder, calibration_images, image_size)

        print("[6/9] Converting quantized encoder")
        quantized_encoder = convert_encoder_after_ptq(prepared_encoder, backend="ethosu")

        print("[7/9] Capturing pre-lowering float and final pre-dequant integer")
        with torch.no_grad():
            quantized_output = quantized_encoder(image).detach().to("cpu", dtype=torch.float32)
        quantized_float_path = output_dir / "host_quantized_pre_lowering_dequant_output.npz"
        save_float_npz(quantized_float_path, quantized_output)

        final_export = torch.export.export(quantized_encoder, (example_input,), strict=True)
        final_export_module = final_export.module()
        with torch.no_grad():
            final_export_output = final_export_module(image).detach().to("cpu", dtype=torch.float32)
        final_export_float_path = output_dir / "host_final_export_pre_lowering_dequant_output.npz"
        save_float_npz(final_export_float_path, final_export_output)

        int_record, all_quantize_records = capture_last_matching_quantize(
            final_export_module,
            image,
            list(final_export_output.shape),
        )
        int_tensor = int_record["value"]
        int_path = output_dir / "host_final_export_pre_lowering_pre_dequant_int_output.npz"
        save_int_npz(int_path, int_tensor)
        pre_lowering_qparams_path = output_dir / "pre_lowering_qparams.csv"
        pre_lowering_qparam_rows = dump_qparams_csv(
            pre_lowering_qparams_path,
            final_export_module,
            "pre_lowering_final_export",
        )

        report["host_quantized_pre_lowering_dequant_output"] = {
            "path": str(quantized_float_path),
            "stats": tensor_stats(quantized_output),
        }
        report["host_final_export_pre_lowering_dequant_output"] = {
            "path": str(final_export_float_path),
            "stats": tensor_stats(final_export_output),
        }
        report["host_final_export_pre_lowering_pre_dequant_int_output"] = {
            "path": str(int_path),
            "node": {k: v for k, v in int_record.items() if k != "value"},
            "stats": tensor_stats(int_tensor),
        }
        report["pre_lowering_qparams_csv"] = {
            "path": str(pre_lowering_qparams_path),
            "row_count": len(pre_lowering_qparam_rows),
        }
        report["captured_quantize_nodes"] = [
            {k: v for k, v in record.items() if k != "value"} for record in all_quantize_records
        ]

        print("[8/9] Lowering same exported quantized graph to Ethos-U")
        report["final_export_graph_summary"] = summarize_fx_graph(final_export, "final_export")
        partitioner = EthosUPartitioner(
            EthosUCompatCompileSpec(
                "ethos-u65-256",
                system_config="Ethos_U65_High_End",
                memory_mode="Dedicated_Sram_384KB",
                config_ini="Arm/vela.ini",
                extra_flags=[],
            )
        )
        edge_manager = to_edge_transform_and_lower(final_export, partitioner=[partitioner])
        report["post_partition_graph_summary"] = summarize_fx_graph(
            edge_manager.exported_program().graph_module,
            "post_partition",
        )
        edge_manager = edge_manager.transform([ReplaceSurvivingQdqWithOutVarPass()])
        post_lowering_graph = edge_manager.exported_program().graph_module
        report["post_partition_qdq_fix_graph_summary"] = summarize_fx_graph(
            post_lowering_graph,
            "post_partition_qdq_fix",
        )
        post_lowering_qparams_path = output_dir / "post_lowering_qparams.csv"
        post_lowering_qparam_rows = dump_qparams_csv(
            post_lowering_qparams_path,
            post_lowering_graph,
            "post_lowering_qdq_fix",
        )
        report["post_lowering_qparams_csv"] = {
            "path": str(post_lowering_qparams_path),
            "row_count": len(post_lowering_qparam_rows),
        }

        print("[9/9] Serializing PTE")
        executorch_program = edge_manager.to_executorch()
        report["runtime_program_summary"] = summarize_executorch_program(executorch_program)
        pte_path = output_dir / args.artifact_name
        pte_path.write_bytes(executorch_program.buffer)
        report["pte_path"] = str(pte_path)
        report["pte_size"] = pte_path.stat().st_size
        report["pte_sha256"] = sha256_path(pte_path)
        report["status"] = "succeeded"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        report["traceback"] = traceback.format_exc()
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
