#!/usr/bin/env python3
"""Build full BHLD PTQ model and prepare VQ eval artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
    describe_ethosu_quantization_profile,
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
        numeric = flat.to(torch.float32)
        stats.update(
            {
                "min": float(numeric.min().item()),
                "max": float(numeric.max().item()),
                "mean": float(numeric.mean().item()),
                "std": float(numeric.std(unbiased=False).item()),
                "first_8": [float(x) for x in numeric[:8].tolist()],
            }
        )
    return stats


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def write_cm33_input_blob(image: torch.Tensor, path: Path) -> dict[str, Any]:
    array = image.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
    payload = array.astype("<f4", copy=False).tobytes()
    header = bytearray(64)
    header[0:8] = b"ETINP001"
    header[8:12] = (len(payload)).to_bytes(4, "little")
    header[12:16] = (array.ndim).to_bytes(4, "little")
    for index, dim in enumerate(array.shape):
        header[16 + 4 * index : 20 + 4 * index] = int(dim).to_bytes(4, "little")
    path.write_bytes(bytes(header) + payload)
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "shape": list(array.shape),
        "dtype": "float32",
        "nbytes": len(payload),
    }


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


def capture_final_int(module: torch.fx.GraphModule, image: torch.Tensor, output_shape: list[int]):
    interp = QuantizeCaptureInterpreter(module)
    _ = interp.run(image)
    matching = [
        record
        for record in interp.quantize_records
        if record["shape"] == output_shape and isinstance(record["value"], torch.Tensor)
    ]
    if not matching:
        raise RuntimeError(f"No quantize_per_tensor node found with output shape {output_shape}")
    return matching[-1]


def parse_final_qparams(record: dict[str, Any]) -> dict[str, Any]:
    args = record["args"]
    return {
        "scale": float(args[1]),
        "zero_point": int(args[2]),
        "qmin": int(args[3]),
        "qmax": int(args[4]),
        "dtype": args[5],
        "node_name": record["name"],
        "node_target": record["target"],
    }


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
    # Global SSIM over RGB channels. This is deterministic and dependency-free.
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


def pair_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.reshape(-1).to(torch.float64)
    bf = b.reshape(-1).to(torch.float64)
    denom = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf)
    diff = (bf - af).abs()
    return {
        "latent_cosine": float(torch.dot(af, bf) / denom) if float(denom) else 0.0,
        "latent_mae": float(diff.mean()),
        "latent_max_abs": float(diff.max()),
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


def summarize_pair(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(rows[0])
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def quantized_dtype_counts(module: torch.fx.GraphModule) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in module.graph.nodes:
        if "quantize_per_tensor" not in str(node.target):
            continue
        if "dequantize_per_tensor" in str(node.target):
            continue
        dtype = str(node.args[5]) if len(node.args) > 5 else "unknown"
        counts[dtype] = counts.get(dtype, 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--titok-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="yucornetto/tokenizer_titok_s128_imagenet")
    parser.add_argument(
        "--encoder-variant",
        default="source_matmul_attention",
        choices=("source_matmul_attention", "source_sdpa_attention"),
        help="Encoder wrapper variant to evaluate.",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--calibration-count", type=int, default=0)
    parser.add_argument("--eval-count", type=int, default=10)
    parser.add_argument("--eval-start", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--artifact-name", default="full_bhld_matmul_full_calib.pte")
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
        default="int8",
        help="Host PTQ profile. int8_surface_a16w8 makes only encoder surface modules A16W8.",
    )
    parser.add_argument(
        "--skip-lowering",
        action="store_true",
        help="Stop after host PTQ/export quality evaluation without producing a PTE.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    board_stage = output_dir / "board_stage"
    board_stage.mkdir(exist_ok=True)
    manifest_path = resolve_input_path(str(args.manifest), REPO_ROOT)
    all_images = load_manifest_records(manifest_path)
    calibration_images = all_images if args.calibration_count <= 0 else all_images[: args.calibration_count]
    eval_images = all_images[args.eval_start : args.eval_start + args.eval_count]
    report_path = output_dir / "host_vq_eval_summary.json"
    report: dict[str, Any] = {
        "status": "started",
        "encoder_variant": args.encoder_variant,
        "quantization_profile": args.quantization_profile,
        "quantization_profile_detail": describe_ethosu_quantization_profile(
            quantization_profile=args.quantization_profile,
        ),
        "skip_lowering": args.skip_lowering,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "calibration_count": len(calibration_images),
        "eval_count": len(eval_images),
        "eval_images": [str(p) for p in eval_images],
        "lpips": "not_computed_lpips_package_not_installed",
    }

    try:
        titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
        titok.requires_grad_(False)
        image_size = int(titok.config.dataset.preprocessing.crop_size)
        report["image_size"] = image_size

        encoder_only, _, _ = build_encoder_quantizer_split(
            titok,
            encoder_variant=args.encoder_variant,
        )
        encoder_only = encoder_only.eval().to("cpu")
        encoder_only.requires_grad_(False)

        example_input = load_image(calibration_images[0], image_size).to("cpu")
        exported_program = export_encoder_program(encoder_only, example_input)
        prepared_encoder, compile_spec = prepare_exported_encoder_for_ptq(
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
        report["compile_spec_flags"] = compile_spec.compiler_flags if compile_spec is not None else None

        calibrate_prepared_encoder(prepared_encoder, calibration_images, image_size)
        quantized_encoder = convert_encoder_after_ptq(prepared_encoder, backend="ethosu")
        final_export = torch.export.export(quantized_encoder, (example_input,), strict=True)
        final_export_module = final_export.module()
        report["final_export_graph_summary"] = summarize_fx_graph(final_export, "final_export")
        report["final_export_quantize_dtype_counts"] = quantized_dtype_counts(final_export_module)

        if not args.skip_lowering:
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
            executorch_program = edge_manager.to_executorch()
            report["runtime_program_summary"] = summarize_executorch_program(executorch_program)
            pte_path = output_dir / args.artifact_name
            pte_path.write_bytes(executorch_program.buffer)
            report["pte_path"] = str(pte_path)
            report["pte_sha256"] = sha256_path(pte_path)
            report["pte_size"] = pte_path.stat().st_size
            (board_stage / pte_path.name).write_bytes(pte_path.read_bytes())

        per_image: list[dict[str, Any]] = []
        fq_rows: list[dict[str, float]] = []
        for index, image_path in enumerate(eval_images):
            image = load_image(image_path, image_size).to("cpu")
            stem = f"{index:03d}_{Path(image_path).stem}"
            with torch.no_grad():
                f_latent = encoder_only(image).detach().to("cpu", dtype=torch.float32)
                q_latent = final_export_module(image).detach().to("cpu", dtype=torch.float32)
            int_record = capture_final_int(final_export_module, image, list(q_latent.shape))
            q_int = int_record["value"].detach().to("cpu")
            qparams = parse_final_qparams(int_record)
            if "final_output_qparams" not in report:
                report["final_output_qparams"] = qparams
            f_tokens, f_top5 = latent_to_topk(titok, f_latent)
            q_tokens, q_top5 = latent_to_topk(titok, q_latent)
            f_decoded = decode_tokens(titok, f_tokens)
            q_decoded = decode_tokens(titok, q_tokens)

            image_dir = output_dir / "host_eval" / stem
            image_dir.mkdir(parents=True, exist_ok=True)
            save_npz(image_dir / "host_outputs.npz",
                f_latent=f_latent.numpy(),
                q_latent=q_latent.numpy(),
                q_int=q_int.numpy(),
                f_tokens=f_tokens.numpy(),
                q_tokens=q_tokens.numpy(),
                f_top5=f_top5.numpy(),
                q_top5=q_top5.numpy(),
                f_decoded=f_decoded.numpy(),
                q_decoded=q_decoded.numpy(),
            )
            input_blob = board_stage / f"{stem}_input_blob.bin"
            input_meta = write_cm33_input_blob(image, input_blob)
            image_pair = pair_metrics(f_latent, q_latent)
            image_pair.update(token_pair_metrics(f_tokens, f_top5, q_tokens, q_top5))
            image_pair.update(
                {
                    "decoded_psnr": psnr(f_decoded, q_decoded),
                    "decoded_ssim": ssim_simple(f_decoded, q_decoded),
                }
            )
            fq_rows.append(image_pair)
            per_image.append(
                {
                    "index": index,
                    "stem": stem,
                    "image": str(image_path),
                    "host_outputs": str(image_dir / "host_outputs.npz"),
                    "board_input_blob": input_meta,
                    "qparams": qparams,
                    "F_vs_Q": image_pair,
                    "F_stats": tensor_stats(f_latent),
                    "Q_stats": tensor_stats(q_latent),
                }
            )

        report["per_image"] = per_image
        report["summary"] = {"F_vs_Q": summarize_pair(fq_rows)}
        report["status"] = "succeeded"
        with (output_dir / "metrics_summary.csv").open("w", newline="") as f:
            fieldnames = ["metric", "F_vs_Q", "Q_vs_B", "F_vs_B"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            fq = report["summary"]["F_vs_Q"]
            for metric in [
                "latent_cosine",
                "latent_mae",
                "vq_token_exact_agreement",
                "vq_top5_agreement_ref_in_candidate",
                "decoded_psnr",
                "decoded_ssim",
            ]:
                writer.writerow({"metric": metric, "F_vs_Q": fq.get(metric), "Q_vs_B": "", "F_vs_B": ""})
            writer.writerow({"metric": "decoded_lpips", "F_vs_Q": "not_available", "Q_vs_B": "", "F_vs_B": ""})
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
