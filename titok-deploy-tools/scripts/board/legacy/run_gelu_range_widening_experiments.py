#!/usr/bin/env python3
"""Run targeted GELU output quantization range-widening experiments."""

from __future__ import annotations

import argparse
import copy
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


TARGET_GELU_INPUT_NODES = ("gelu_1", "gelu_2", "gelu_5", "gelu_6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--titok-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="yucornetto/tokenizer_titok_s128_imagenet")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--calibration-count", type=int, default=500)
    parser.add_argument("--eval-count", type=int, default=10)
    parser.add_argument("--eval-start", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--multipliers", type=float, nargs="+", default=[1.25, 1.5, 2.0])
    parser.add_argument(
        "--quantization-profile",
        default="int8_surface_transformer_norm_residual_a16w8",
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
    )
    return parser.parse_args()


def _is_quantize_node(node: Node) -> bool:
    target = str(node.target)
    return "quantize_per_tensor" in target and "dequantize_per_tensor" not in target


def _is_dequantize_node(node: Node) -> bool:
    return "dequantize_per_tensor" in str(node.target)


def _as_number(value: Any) -> float | int:
    if isinstance(value, torch.Tensor):
        return value.item()
    return value


def _patch_args_scale(node: Node, multiplier: float) -> None:
    args = list(node.args)
    args[1] = float(_as_number(args[1])) * multiplier
    node.args = tuple(args)


def target_gelu_quantize_nodes(module: torch.fx.GraphModule) -> list[Node]:
    nodes: list[Node] = []
    for node in module.graph.nodes:
        if not _is_quantize_node(node):
            continue
        if not node.args or not isinstance(node.args[0], Node):
            continue
        input_node = node.args[0]
        if input_node.name in TARGET_GELU_INPUT_NODES and str(input_node.target) == "aten.gelu.default":
            nodes.append(node)
    return nodes


def widen_target_gelu_scales(module: torch.fx.GraphModule, multiplier: float) -> list[dict[str, Any]]:
    patched: list[dict[str, Any]] = []
    for quant_node in target_gelu_quantize_nodes(module):
        original_scale = float(_as_number(quant_node.args[1]))
        _patch_args_scale(quant_node, multiplier)
        dequant_users = [user for user in quant_node.users if _is_dequantize_node(user)]
        for dequant_node in dequant_users:
            _patch_args_scale(dequant_node, multiplier)
        patched.append(
            {
                "input_node": quant_node.args[0].name,
                "quant_node": quant_node.name,
                "original_scale": original_scale,
                "new_scale": original_scale * multiplier,
                "patched_dequant_nodes": [node.name for node in dequant_users],
            }
        )
    module.graph.lint()
    module.recompile()
    return patched


def latent_to_topk(titok, latent: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    z = latent.detach().to("cpu", dtype=torch.float32)
    z = z.permute(0, 2, 3, 1).contiguous()
    flat = z.reshape(-1, z.shape[-1])
    if titok.quantize.use_l2_norm:
        flat = torch.nn.functional.normalize(flat, dim=-1)
        embedding = torch.nn.functional.normalize(titok.quantize.embedding.weight.detach().to("cpu"), dim=-1)
    else:
        embedding = titok.quantize.embedding.weight.detach().to("cpu")
    distances = (
        torch.sum(flat**2, dim=1, keepdim=True)
        + torch.sum(embedding**2, dim=1)
        - 2 * torch.matmul(flat, embedding.t())
    )
    topk = torch.topk(distances, k=k, largest=False, dim=1).indices
    return topk[:, 0].reshape(latent.shape[0], latent.shape[2], latent.shape[3]), topk.reshape(
        latent.shape[0], latent.shape[2], latent.shape[3], k
    )


def decode_tokens(titok, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.clamp(titok.decode_tokens(tokens.reshape(tokens.shape[0], 1, -1)), 0.0, 1.0)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def ssim_simple(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().to(torch.float64)
    y = b.detach().to(torch.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    values = []
    for channel in range(x.shape[1]):
        xc = x[:, channel].reshape(-1)
        yc = y[:, channel].reshape(-1)
        mux = xc.mean()
        muy = yc.mean()
        vx = ((xc - mux) ** 2).mean()
        vy = ((yc - muy) ** 2).mean()
        cov = ((xc - mux) * (yc - muy)).mean()
        values.append(float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2))))
    return float(sum(values) / len(values))


def pair_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.reshape(-1).to(torch.float64)
    cand = candidate.reshape(-1).to(torch.float64)
    denom = torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(cand)
    diff = (cand - ref).abs()
    return {
        "latent_cosine": float(torch.dot(ref, cand) / denom) if float(denom) else 0.0,
        "latent_mae": float(diff.mean()),
        "latent_max_abs": float(diff.max()),
    }


def image_reconstruction_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    diff = (candidate.detach().to(torch.float32) - reference.detach().to(torch.float32)).abs()
    mse = torch.mean(diff * diff).item()
    return {
        "input_psnr": psnr(reference, candidate),
        "input_ssim": ssim_simple(reference, candidate),
        "input_mae": float(diff.mean().item()),
        "input_rmse": math.sqrt(mse),
    }


def token_pair_metrics(ref_tokens: torch.Tensor, ref_top5: torch.Tensor, cand_tokens: torch.Tensor, cand_top5: torch.Tensor) -> dict[str, float]:
    ref_flat = ref_tokens.reshape(-1)
    cand_flat = cand_tokens.reshape(-1)
    exact = (ref_flat == cand_flat).to(torch.float32)
    ref_in_cand_top5 = (cand_top5.reshape(-1, cand_top5.shape[-1]) == ref_flat[:, None]).any(dim=1).to(torch.float32)
    cand_in_ref_top5 = (ref_top5.reshape(-1, ref_top5.shape[-1]) == cand_flat[:, None]).any(dim=1).to(torch.float32)
    return {
        "vq_token_exact_agreement": float(exact.mean()),
        "vq_top5_agreement_ref_in_candidate": float(ref_in_cand_top5.mean()),
        "vq_top5_agreement_candidate_in_ref": float(cand_in_ref_top5.mean()),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(rows[0])
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


class TargetSaturationInterpreter(Interpreter):
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
        if input_node.name not in TARGET_GELU_INPUT_NODES:
            return result
        source = self.env.get(input_node)
        if not isinstance(source, torch.Tensor):
            return result
        scale = float(_as_number(node.args[1]))
        zero_point = int(_as_number(node.args[2]))
        qmin = int(_as_number(node.args[3]))
        qmax = int(_as_number(node.args[4]))
        float_value = source.detach().to("cpu", dtype=torch.float32)
        q_numeric = result.detach().to("cpu").to(torch.int32)
        dequant_min = (qmin - zero_point) * scale
        dequant_max = (qmax - zero_point) * scale
        flat = float_value.reshape(-1)
        self.records.append(
            {
                "input_node": input_node.name,
                "quant_node": node.name,
                "scale": scale,
                "zero_point": zero_point,
                "qmin": qmin,
                "qmax": qmax,
                "dequant_min": dequant_min,
                "dequant_max": dequant_max,
                "float_min": float(flat.min().item()),
                "float_max": float(flat.max().item()),
                "quantized_min": int(q_numeric.min().item()),
                "quantized_max": int(q_numeric.max().item()),
                "pct_at_qmin": float((q_numeric == qmin).to(torch.float32).mean().item()),
                "pct_at_qmax": float((q_numeric == qmax).to(torch.float32).mean().item()),
                "pct_below_dequant_min": float((flat < dequant_min).to(torch.float32).mean().item()),
                "pct_above_dequant_max": float((flat > dequant_max).to(torch.float32).mean().item()),
                "numel": int(flat.numel()),
            }
        )
        return result


def merge_saturation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    weights = np.array([row["numel"] for row in rows], dtype=np.float64)
    weights = weights / weights.sum()
    merged = {
        "input_node": first["input_node"],
        "quant_node": first["quant_node"],
        "scale": first["scale"],
        "zero_point": first["zero_point"],
        "qmin": first["qmin"],
        "qmax": first["qmax"],
        "dequant_min": first["dequant_min"],
        "dequant_max": first["dequant_max"],
        "float_min": min(row["float_min"] for row in rows),
        "float_max": max(row["float_max"] for row in rows),
        "quantized_min": min(row["quantized_min"] for row in rows),
        "quantized_max": max(row["quantized_max"] for row in rows),
    }
    for key in ("pct_at_qmin", "pct_at_qmax", "pct_below_dequant_min", "pct_above_dequant_max"):
        values = np.array([row[key] for row in rows], dtype=np.float64)
        merged[key] = float(np.sum(values * weights))
    merged["pct_at_extreme"] = merged["pct_at_qmin"] + merged["pct_at_qmax"]
    merged["pct_outside_dequant_range"] = merged["pct_below_dequant_min"] + merged["pct_above_dequant_max"]
    return merged


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    output_dir = resolve_output_dir(REPO_ROOT, str(args.output_dir))
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
        "target_gelu_input_nodes": list(TARGET_GELU_INPUT_NODES),
        "multipliers": args.multipliers,
        "manifest": str(manifest_path),
        "calibration_count": len(calibration_images),
        "eval_count": len(eval_images),
        "eval_images": [str(path) for path in eval_images],
    }
    summary_path = output_dir / "gelu_range_widening_summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n")

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
        base_module = final_export.module()
        base_targets = [
            {
                "input_node": node.args[0].name,
                "quant_node": node.name,
                "scale": float(_as_number(node.args[1])),
                "zero_point": int(_as_number(node.args[2])),
                "qmin": int(_as_number(node.args[3])),
                "qmax": int(_as_number(node.args[4])),
            }
            for node in target_gelu_quantize_nodes(base_module)
        ]
        if sorted(row["input_node"] for row in base_targets) != sorted(TARGET_GELU_INPUT_NODES):
            raise RuntimeError(f"Did not find exactly requested GELU quant nodes: {base_targets}")
        report["base_target_quant_nodes"] = base_targets

        experiments: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        saturation_rows: list[dict[str, Any]] = []
        for multiplier in args.multipliers:
            module = copy.deepcopy(base_module)
            patched = widen_target_gelu_scales(module, multiplier)
            exp_dir = output_dir / f"scale_x{multiplier:g}".replace(".", "p")
            exp_dir.mkdir(parents=True, exist_ok=True)

            metric_rows: list[dict[str, float]] = []
            float_input_metric_rows: list[dict[str, float]] = []
            quant_input_metric_rows: list[dict[str, float]] = []
            per_image: list[dict[str, Any]] = []
            grouped_sat: dict[str, list[dict[str, Any]]] = {}
            for index, image_path in enumerate(eval_images):
                image = load_image(image_path, image_size).to("cpu")
                stem = f"{index:03d}_{Path(image_path).stem}"
                with torch.no_grad():
                    f_latent = encoder_only(image).detach().to("cpu", dtype=torch.float32)
                    interp = TargetSaturationInterpreter(module)
                    q_latent = interp.run(image).detach().to("cpu", dtype=torch.float32)
                for sat_row in interp.records:
                    sat_row["multiplier"] = multiplier
                    sat_row["image_index"] = index
                    sat_row["image"] = str(image_path)
                    saturation_rows.append(sat_row)
                    grouped_sat.setdefault(sat_row["input_node"], []).append(sat_row)

                f_tokens, f_top5 = latent_to_topk(titok, f_latent)
                q_tokens, q_top5 = latent_to_topk(titok, q_latent)
                f_decoded = decode_tokens(titok, f_tokens)
                q_decoded = decode_tokens(titok, q_tokens)
                metrics = pair_metrics(f_latent, q_latent)
                metrics.update(token_pair_metrics(f_tokens, f_top5, q_tokens, q_top5))
                metrics.update({"decoded_psnr": psnr(f_decoded, q_decoded), "decoded_ssim": ssim_simple(f_decoded, q_decoded)})
                float_input_metrics = image_reconstruction_metrics(image, f_decoded)
                quant_input_metrics = image_reconstruction_metrics(image, q_decoded)
                metric_rows.append(metrics)
                float_input_metric_rows.append(float_input_metrics)
                quant_input_metric_rows.append(quant_input_metrics)
                per_image.append(
                    {
                        "index": index,
                        "stem": stem,
                        "image": str(image_path),
                        "F_vs_Q": metrics,
                        "input_vs_float": float_input_metrics,
                        "input_vs_quantized": quant_input_metrics,
                    }
                )

            metric_summary = summarize(metric_rows)
            float_input_summary = summarize(float_input_metric_rows)
            quant_input_summary = summarize(quant_input_metric_rows)
            sat_summary = {name: merge_saturation(rows) for name, rows in sorted(grouped_sat.items())}
            experiment = {
                "multiplier": multiplier,
                "patched_nodes": patched,
                "summary": {
                    "F_vs_Q": metric_summary,
                    "input_vs_float": float_input_summary,
                    "input_vs_quantized": quant_input_summary,
                    "gelu_saturation": sat_summary,
                },
                "per_image": per_image,
            }
            experiments.append(experiment)
            (exp_dir / "summary.json").write_text(json.dumps(experiment, indent=2) + "\n")

            summary_row: dict[str, Any] = {
                "multiplier": multiplier,
                "latent_cosine": metric_summary["latent_cosine"],
                "latent_mae": metric_summary["latent_mae"],
                "vq_exact": metric_summary["vq_token_exact_agreement"],
                "vq_top5_ref_in_candidate": metric_summary["vq_top5_agreement_ref_in_candidate"],
                "vq_top5_candidate_in_ref": metric_summary["vq_top5_agreement_candidate_in_ref"],
                "decoded_psnr": metric_summary["decoded_psnr"],
                "decoded_ssim": metric_summary["decoded_ssim"],
                "float_input_psnr": float_input_summary["input_psnr"],
                "float_input_ssim": float_input_summary["input_ssim"],
                "float_input_mae": float_input_summary["input_mae"],
                "float_input_rmse": float_input_summary["input_rmse"],
                "quant_input_psnr": quant_input_summary["input_psnr"],
                "quant_input_ssim": quant_input_summary["input_ssim"],
                "quant_input_mae": quant_input_summary["input_mae"],
                "quant_input_rmse": quant_input_summary["input_rmse"],
            }
            for gelu_name in TARGET_GELU_INPUT_NODES:
                sat = sat_summary[gelu_name]
                summary_row[f"{gelu_name}_outside_pct"] = 100.0 * sat["pct_outside_dequant_range"]
                summary_row[f"{gelu_name}_at_extreme_pct"] = 100.0 * sat["pct_at_extreme"]
                summary_row[f"{gelu_name}_scale"] = sat["scale"]
            summary_rows.append(summary_row)

        write_csv(output_dir / "gelu_range_widening_summary.csv", summary_rows)
        write_csv(output_dir / "gelu_range_widening_saturation_per_image.csv", saturation_rows)

        md = [
            "# GELU Output Range Widening",
            "",
            f"Profile: `{args.quantization_profile}`",
            f"Target GELU outputs: `{', '.join(TARGET_GELU_INPUT_NODES)}`",
            f"Calibration images: {len(calibration_images)}",
            f"Eval images: {len(eval_images)}",
            "",
            "| Scale multiplier | gelu_1 outside % | gelu_2 outside % | gelu_5 outside % | gelu_6 outside % | VQ exact | VQ top5 | Latent cosine | Latent MAE | PSNR | SSIM |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in summary_rows:
            md.append(
                f"| {row['multiplier']:.2f} | "
                f"{row['gelu_1_outside_pct']:.4f} | "
                f"{row['gelu_2_outside_pct']:.4f} | "
                f"{row['gelu_5_outside_pct']:.4f} | "
                f"{row['gelu_6_outside_pct']:.4f} | "
                f"{row['vq_exact']:.6f} | "
                f"{row['vq_top5_ref_in_candidate']:.6f} | "
                f"{row['latent_cosine']:.6f} | "
                f"{row['latent_mae']:.6f} | "
                f"{row['decoded_psnr']:.4f} | "
                f"{row['decoded_ssim']:.4f} |"
            )
        (output_dir / "gelu_range_widening_report.md").write_text("\n".join(md) + "\n")

        recon_md = [
            "# Input Reconstruction Metrics",
            "",
            "Metrics compare decoded reconstructions directly against the input image crop.",
            "",
            "| Scale multiplier | Float PSNR | Float SSIM | Float MAE | Float RMSE | Quant PSNR | Quant SSIM | Quant MAE | Quant RMSE |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in summary_rows:
            recon_md.append(
                f"| {row['multiplier']:.2f} | "
                f"{row['float_input_psnr']:.4f} | "
                f"{row['float_input_ssim']:.4f} | "
                f"{row['float_input_mae']:.6f} | "
                f"{row['float_input_rmse']:.6f} | "
                f"{row['quant_input_psnr']:.4f} | "
                f"{row['quant_input_ssim']:.4f} | "
                f"{row['quant_input_mae']:.6f} | "
                f"{row['quant_input_rmse']:.6f} |"
            )
        (output_dir / "input_reconstruction_metrics.md").write_text("\n".join(recon_md) + "\n")

        report.update(
            {
                "status": "succeeded",
                "experiments": experiments,
                "summary_csv": str(output_dir / "gelu_range_widening_summary.csv"),
                "report_md": str(output_dir / "gelu_range_widening_report.md"),
                "input_reconstruction_report_md": str(output_dir / "input_reconstruction_metrics.md"),
                "saturation_per_image_csv": str(output_dir / "gelu_range_widening_saturation_per_image.csv"),
            }
        )
        summary_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": "succeeded", "summary_rows": summary_rows, "report_md": report["report_md"]}, indent=2))
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        summary_path.write_text(json.dumps(report, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
