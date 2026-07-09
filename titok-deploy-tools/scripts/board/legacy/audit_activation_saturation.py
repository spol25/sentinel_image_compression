#!/usr/bin/env python3
"""Audit activation saturation at PT2E quantize_per_tensor nodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.fx import Interpreter, Node

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

from titok_deploy_tools.ptq_tools.ptq import (  # noqa: E402
    build_encoder_quantizer_split,
    calibrate_prepared_encoder,
    convert_encoder_after_ptq,
    describe_ethosu_quantization_profile,
    export_encoder_program,
    load_manifest_records,
    prepare_exported_encoder_for_ptq,
)
from titok_deploy_tools.wrapper_tools.titok_env import add_titok_root_to_path  # noqa: E402
from titok_deploy_tools.wrapper_tools.utils import load_image, resolve_input_path, resolve_output_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--titok-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="yucornetto/tokenizer_titok_s128_imagenet")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--calibration-count", type=int, default=500)
    parser.add_argument("--eval-count", type=int, default=10)
    parser.add_argument("--eval-start", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--quantization-profile",
        choices=(
            "int8",
            "a16w8",
            "int8_surface_a16w8",
            "int8_surface_transformer_norm_a16w8",
            "int8_surface_transformer_norm_residual_a16w8",
            "int8_surface_transformer_norm_residual_mlp_output_a16w8",
            "int8_surface_transformer_norm_residual_mlp_output_boundary_a16w8",
            "int8_surface_transformer_norm_residual_mlp_gelu_a16w8",
            "int8_surface_transformer_norm_residual_post_gelu_boundary_a16w8",
        ),
        default="int8_surface_transformer_norm_residual_a16w8",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Number of high-saturation and high-MAE rows to include in Markdown.",
    )
    return parser.parse_args()


def _meta_stack(node: Node) -> str:
    stack = node.meta.get("nn_module_stack")
    if not stack:
        return ""
    parts: list[str] = []
    for value in stack.values():
        if isinstance(value, tuple) and value:
            parts.append(str(value[0]))
        else:
            parts.append(str(value))
    return " > ".join(parts)


def _shape(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    return None


def _as_number(value: Any) -> float | int:
    if isinstance(value, torch.Tensor):
        return value.item()
    return value


def _dtype_name(value: Any) -> str:
    text = str(value)
    return text.replace("torch.", "")


def _is_quantize_node(node: Node) -> bool:
    target = str(node.target)
    return "quantize_per_tensor" in target and "dequantize_per_tensor" not in target


def _should_focus(record: dict[str, Any]) -> bool:
    text = " ".join(
        str(record.get(key, ""))
        for key in (
            "node_name",
            "input_node",
            "input_target",
            "user_nodes",
            "user_targets",
            "module_stack",
            "source_fn_stack",
        )
    ).lower()
    shape = record.get("shape") or []
    if any(
        token in text
        for token in (
            "transformer",
            "attention",
            "attn",
            "mlp",
            "gelu",
            "residual",
            "bmm",
            "matmul",
            "linear",
            "add",
            "layer_norm",
        )
    ):
        return True
    return (
        shape in ([1, 129, 512], [1, 129, 2048], [1, 8, 129, 64], [1, 8, 129, 129], [1, 12, 1, 128])
        or (len(shape) >= 3 and 129 in shape)
    )


class SaturationInterpreter(Interpreter):
    def __init__(self, module: torch.fx.GraphModule):
        super().__init__(module)
        self.records: list[dict[str, Any]] = []

    def run_node(self, node):  # noqa: ANN001
        result = super().run_node(node)
        if not _is_quantize_node(node) or not isinstance(result, torch.Tensor):
            return result
        if not node.args or not isinstance(node.args[0], Node):
            return result

        input_node = node.args[0]
        source = self.env.get(input_node)
        if not isinstance(source, torch.Tensor):
            return result

        scale = float(_as_number(node.args[1]))
        zero_point = int(_as_number(node.args[2]))
        qmin = int(_as_number(node.args[3]))
        qmax = int(_as_number(node.args[4]))
        dtype = _dtype_name(node.args[5]) if len(node.args) > 5 else str(result.dtype)

        float_value = source.detach().to("cpu", dtype=torch.float32)
        q_value = result.detach().to("cpu")
        q_numeric = q_value.to(torch.int32)
        dequantized = (q_numeric.to(torch.float32) - zero_point) * scale
        dequant_min = (qmin - zero_point) * scale
        dequant_max = (qmax - zero_point) * scale
        flat = float_value.reshape(-1)
        deq_flat = dequantized.reshape(-1)
        diff = deq_flat - flat
        denom = torch.linalg.vector_norm(flat) * torch.linalg.vector_norm(deq_flat)
        cosine = float(torch.dot(flat, deq_flat) / denom) if float(denom) else 0.0

        users = list(node.users)
        self.records.append(
            {
                "node_name": node.name,
                "node_target": str(node.target),
                "input_node": input_node.name,
                "input_target": str(input_node.target),
                "user_nodes": ",".join(user.name for user in users),
                "user_targets": ",".join(str(user.target) for user in users),
                "module_stack": _meta_stack(node),
                "source_fn_stack": str(node.meta.get("source_fn_stack", "")),
                "shape": _shape(float_value),
                "numel": int(float_value.numel()),
                "dtype": dtype,
                "scale": scale,
                "zero_point": zero_point,
                "qmin": qmin,
                "qmax": qmax,
                "dequant_min": dequant_min,
                "dequant_max": dequant_max,
                "float_min": float(flat.min().item()),
                "float_max": float(flat.max().item()),
                "float_mean": float(flat.mean().item()),
                "float_std": float(flat.std(unbiased=False).item()),
                "quantized_min": int(q_numeric.min().item()),
                "quantized_max": int(q_numeric.max().item()),
                "pct_at_qmin": float((q_numeric == qmin).to(torch.float32).mean().item()),
                "pct_at_qmax": float((q_numeric == qmax).to(torch.float32).mean().item()),
                "pct_below_dequant_min": float((flat < dequant_min).to(torch.float32).mean().item()),
                "pct_above_dequant_max": float((flat > dequant_max).to(torch.float32).mean().item()),
                "quantization_mae": float(diff.abs().mean().item()),
                "quantization_max_abs": float(diff.abs().max().item()),
                "quantization_cosine": cosine,
                "focus": False,
            }
        )
        self.records[-1]["focus"] = _should_focus(self.records[-1])
        return result


def _merge_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    weights = np.array([row["numel"] for row in rows], dtype=np.float64)
    weights = weights / weights.sum()
    merged = {
        "node_name": first["node_name"],
        "node_target": first["node_target"],
        "input_node": first["input_node"],
        "input_target": first["input_target"],
        "user_nodes": first["user_nodes"],
        "user_targets": first["user_targets"],
        "module_stack": first["module_stack"],
        "source_fn_stack": first["source_fn_stack"],
        "shape": first["shape"],
        "numel_per_image": first["numel"],
        "dtype": first["dtype"],
        "scale": first["scale"],
        "zero_point": first["zero_point"],
        "qmin": first["qmin"],
        "qmax": first["qmax"],
        "dequant_min": first["dequant_min"],
        "dequant_max": first["dequant_max"],
        "focus": bool(first["focus"]),
        "eval_images": len(rows),
        "float_min": min(row["float_min"] for row in rows),
        "float_max": max(row["float_max"] for row in rows),
        "quantized_min": min(row["quantized_min"] for row in rows),
        "quantized_max": max(row["quantized_max"] for row in rows),
    }
    for key in (
        "float_mean",
        "float_std",
        "pct_at_qmin",
        "pct_at_qmax",
        "pct_below_dequant_min",
        "pct_above_dequant_max",
        "quantization_mae",
        "quantization_max_abs",
        "quantization_cosine",
    ):
        values = np.array([row[key] for row in rows], dtype=np.float64)
        if key == "quantization_max_abs":
            merged[key] = float(values.max())
        else:
            merged[key] = float(np.sum(values * weights))
    merged["pct_at_extreme"] = merged["pct_at_qmin"] + merged["pct_at_qmax"]
    merged["pct_outside_dequant_range"] = merged["pct_below_dequant_min"] + merged["pct_above_dequant_max"]
    return merged


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _short_name(row: dict[str, Any]) -> str:
    module = row.get("module_stack") or ""
    if module:
        return module.split(" > ")[-1]
    return f"{row['input_node']} -> {row['node_name']}"


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Node | Shape | DType | Float min/max | Dequant min/max | Q min/max | % qmin | % qmax | % outside | MAE | Cosine | Users |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_short_name(row)}`<br>`{row['node_name']}`",
                    f"`{row['shape']}`",
                    f"`{row['dtype']}`",
                    f"{row['float_min']:.5g} / {row['float_max']:.5g}",
                    f"{row['dequant_min']:.5g} / {row['dequant_max']:.5g}",
                    f"{row['quantized_min']} / {row['quantized_max']}",
                    f"{100.0 * row['pct_at_qmin']:.4f}",
                    f"{100.0 * row['pct_at_qmax']:.4f}",
                    f"{100.0 * row['pct_outside_dequant_range']:.4f}",
                    f"{row['quantization_mae']:.6g}",
                    f"{row['quantization_cosine']:.6f}",
                    f"`{row['user_targets'][:90]}`",
                ]
            )
            + " |"
        )
    return lines


def main() -> None:
    args = parse_args()
    add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_input_path(str(args.manifest), REPO_ROOT)
    all_images = load_manifest_records(manifest_path)
    calibration_images = all_images[: args.calibration_count] if args.calibration_count > 0 else all_images
    eval_images = all_images[args.eval_start : args.eval_start + args.eval_count]
    report: dict[str, Any] = {
        "status": "started",
        "quantization_profile": args.quantization_profile,
        "quantization_profile_detail": describe_ethosu_quantization_profile(
            quantization_profile=args.quantization_profile,
        ),
        "manifest": str(manifest_path),
        "calibration_count": len(calibration_images),
        "eval_count": len(eval_images),
        "eval_images": [str(path) for path in eval_images],
    }
    (output_dir / "activation_saturation_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
        titok.requires_grad_(False)
        image_size = int(titok.config.dataset.preprocessing.crop_size)
        report["image_size"] = image_size

        encoder_only, _, _ = build_encoder_quantizer_split(
            titok,
            encoder_variant="source_matmul_attention",
        )
        encoder_only = encoder_only.eval().to("cpu")
        encoder_only.requires_grad_(False)

        example_input = load_image(calibration_images[0], image_size).to("cpu")
        exported_program = export_encoder_program(encoder_only, example_input)
        prepared_encoder, _ = prepare_exported_encoder_for_ptq(
            exported_program,
            backend="ethosu",
            is_per_channel=True,
            quantization_profile=args.quantization_profile,
            ethos_target="ethos-u65-256",
            ethos_system_config="Ethos_U65_High_End",
            ethos_memory_mode="Dedicated_Sram_384KB",
            ethos_config_ini="Arm/vela.ini",
            ethos_extra_flags=[],
            quantize_matmul=False,
        )
        calibrate_prepared_encoder(prepared_encoder, calibration_images, image_size)
        quantized_encoder = convert_encoder_after_ptq(prepared_encoder, backend="ethosu")
        final_export = torch.export.export(quantized_encoder, (example_input,), strict=True)
        final_export_module = final_export.module()

        per_image_rows: list[dict[str, Any]] = []
        grouped: dict[str, list[dict[str, Any]]] = {}
        for image_index, image_path in enumerate(eval_images):
            image = load_image(image_path, image_size).to("cpu")
            interp = SaturationInterpreter(final_export_module)
            with torch.no_grad():
                interp.run(image)
            for row in interp.records:
                row["image_index"] = image_index
                row["image"] = str(image_path)
                per_image_rows.append(row)
                grouped.setdefault(row["node_name"], []).append(row)

        aggregate_rows = [_merge_stats(rows) for _, rows in sorted(grouped.items())]
        aggregate_rows.sort(key=lambda row: row["pct_at_extreme"], reverse=True)
        focus_rows = [row for row in aggregate_rows if row["focus"]]
        focus_rows.sort(key=lambda row: row["pct_at_extreme"], reverse=True)
        high_mae_focus_rows = sorted(focus_rows, key=lambda row: row["quantization_mae"], reverse=True)

        _write_csv(output_dir / "activation_saturation_per_image.csv", per_image_rows)
        _write_csv(output_dir / "activation_saturation_aggregate.csv", aggregate_rows)
        _write_csv(output_dir / "activation_saturation_focus.csv", focus_rows)

        report.update(
            {
                "status": "succeeded",
                "num_quantize_nodes": len(aggregate_rows),
                "num_focus_quantize_nodes": len(focus_rows),
                "max_pct_at_extreme": aggregate_rows[0]["pct_at_extreme"] if aggregate_rows else None,
                "max_focus_pct_at_extreme": focus_rows[0]["pct_at_extreme"] if focus_rows else None,
                "max_focus_quantization_mae": high_mae_focus_rows[0]["quantization_mae"] if high_mae_focus_rows else None,
                "csv": {
                    "per_image": str(output_dir / "activation_saturation_per_image.csv"),
                    "aggregate": str(output_dir / "activation_saturation_aggregate.csv"),
                    "focus": str(output_dir / "activation_saturation_focus.csv"),
                },
            }
        )
        (output_dir / "activation_saturation_summary.json").write_text(json.dumps(report, indent=2) + "\n")

        md: list[str] = [
            "# Activation Saturation Audit",
            "",
            f"Profile: `{args.quantization_profile}`",
            f"Calibration images: {len(calibration_images)}",
            f"Eval images: {len(eval_images)}",
            f"Quantize nodes audited: {len(aggregate_rows)}",
            f"Focus quantize nodes: {len(focus_rows)}",
            "",
            "Metrics are averaged across eval images, weighted by tensor element count. `% qmin` and `% qmax` are fractions of quantized activation values exactly at the quantization extrema.",
            "",
            "## Highest Saturation: Focus Nodes",
            "",
        ]
        md.extend(_markdown_table(focus_rows[: args.top_k]))
        md.extend(["", "## Highest Quantization MAE: Focus Nodes", ""])
        md.extend(_markdown_table(high_mae_focus_rows[: args.top_k]))
        md.extend(["", "## Highest Saturation: All Nodes", ""])
        md.extend(_markdown_table(aggregate_rows[: args.top_k]))
        (output_dir / "activation_saturation_report.md").write_text("\n".join(md) + "\n")
        print(json.dumps(report, indent=2))
    except Exception as exc:  # pragma: no cover - diagnostic script
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        (output_dir / "activation_saturation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
