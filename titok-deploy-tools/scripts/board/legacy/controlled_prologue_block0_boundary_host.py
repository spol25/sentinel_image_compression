#!/usr/bin/env python3
"""Export/lower prologue/block0 boundary probes for board comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
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
    convert_encoder_after_ptq,
    export_encoder_program,
    load_manifest_records,
    prepare_exported_encoder_for_ptq,
)
from titok_deploy_tools.wrapper_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.wrapper_tools.utils import load_image, resolve_input_path, resolve_output_dir


INPUT_MAGIC = b"ETINP001"
INPUT_HEADER_SIZE = 64
PROBES = ("prologue_out", "block0_in_forced")


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
    if value.dtype not in (torch.int8, torch.uint8):
        value = value.to(torch.int16)
    np.savez(path, output_int=value.numpy())


def save_input_blob(path: Path, metadata_path: Path, tensor: torch.Tensor, source: str) -> None:
    array = tensor.detach().to("cpu", dtype=torch.float32).contiguous().numpy().astype("<f4", copy=False)
    payload = array.tobytes(order="C")
    header = bytearray(INPUT_HEADER_SIZE)
    header[: len(INPUT_MAGIC)] = INPUT_MAGIC
    struct.pack_into("<I", header, 8, len(payload))
    struct.pack_into("<I", header, 12, array.ndim)
    for i, dim in enumerate(array.shape[:6]):
        struct.pack_into("<I", header, 16 + 4 * i, int(dim))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + payload)
    metadata_path.write_text(
        json.dumps(
            {
                "source": source,
                "output": str(path),
                "magic": INPUT_MAGIC.decode("ascii"),
                "header_size": INPUT_HEADER_SIZE,
                "payload_size": len(payload),
                "dtype": "float32",
                "shape": list(array.shape),
                "min": float(array.min()),
                "max": float(array.max()),
                "mean": float(array.mean()),
                "sha256": sha256_path(path),
            },
            indent=2,
        )
        + "\n"
    )


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    af = a.astype(np.float32).reshape(-1)
    bf = b.astype(np.float32).reshape(-1)
    d = bf - af
    ad = np.abs(d)
    return {
        "exact_equal": bool(np.array_equal(a, b)),
        "exact_match_count": int(np.sum(a.reshape(-1) == b.reshape(-1))),
        "numel": int(af.size),
        "max_abs_error": float(ad.max()) if ad.size else 0.0,
        "mean_abs_error": float(ad.mean()) if ad.size else 0.0,
        "rmse": float(math.sqrt(float(np.mean(d * d)))) if d.size else 0.0,
        "cosine_similarity": float(
            np.dot(af, bf) / max(float(np.linalg.norm(af) * np.linalg.norm(bf)), 1e-12)
        )
        if af.size
        else 1.0,
    }


def _shape_from_node(node) -> str:  # noqa: ANN001
    value = node.meta.get("val") if hasattr(node, "meta") else None
    if isinstance(value, torch.Tensor):
        return "x".join(str(dim) for dim in value.shape)
    return ""


def _node_arg_value(arg, module: torch.fx.GraphModule):  # noqa: ANN001
    if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
        value = module
        for part in str(arg.target).split("."):
            value = getattr(value, part)
        return value
    return arg


def _qparam_scalar_or_stats(value) -> dict[str, str]:  # noqa: ANN001
    if isinstance(value, torch.Tensor):
        detached = value.detach().to("cpu")
        flat = detached.reshape(-1)
        result = {"value": "", "shape": "x".join(str(dim) for dim in detached.shape), "min": "", "max": "", "first_8": ""}
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
    return {"value": str(value), "shape": "", "min": "", "max": "", "first_8": ""}


def dump_qparams_csv(path: Path, module: torch.fx.GraphModule, stage: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for node in module.graph.nodes:
        target = str(node.target)
        if not any(token in target for token in ("quantize_per_tensor", "dequantize_per_tensor", "quantize_per_channel", "dequantize_per_channel")):
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
    fieldnames = list(rows[0].keys()) if rows else [
        "stage", "node_name", "op_kind", "qscheme", "target", "tensor_shape",
        "scale", "scale_shape", "scale_min", "scale_max", "scale_first_8",
        "zero_point", "zero_point_shape", "zero_point_min", "zero_point_max",
        "zero_point_first_8", "axis", "qmin", "qmax", "dtype", "args",
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
        if "quantize_per_tensor" in target and "dequantize_per_tensor" not in target and isinstance(result, torch.Tensor):
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


def capture_last_matching_quantize(module: torch.fx.GraphModule, tensor: torch.Tensor, output_shape: list[int]):
    interp = QuantizeCaptureInterpreter(module)
    result = interp.run(tensor)
    if isinstance(result, tuple):
        result = result[0]
    matching = [
        record
        for record in interp.quantize_records
        if record["shape"] == output_shape and isinstance(record["value"], torch.Tensor)
    ]
    if not matching:
        raise RuntimeError(f"No quantize_per_tensor node found with output shape {output_shape}")
    return matching[-1], interp.quantize_records


class BoundaryProbe(torch.nn.Module):
    def __init__(self, titok, probe: str, channel_start: int, channel_count: int):
        super().__init__()
        if probe not in PROBES:
            raise ValueError(f"Unsupported probe {probe!r}")
        self.probe = probe
        self.encoder = titok.encoder
        self.block = titok.encoder.transformer[0]
        self.channel_start = channel_start
        self.channel_count = channel_count
        self.latent_token_start = 1 + titok.encoder.grid_size ** 2
        self.register_parameter("latent_tokens", titok.latent_tokens)

    def _prologue(self, pixel_values: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        batch_size = pixel_values.shape[0]
        x = encoder.patch_embed(pixel_values)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        class_embedding = encoder.class_embedding.unsqueeze(0).expand(batch_size, -1, -1).to(x.dtype)
        x = torch.cat([class_embedding, x], dim=1)
        x = x + encoder.positional_embedding.to(x.dtype)
        latent_tokens = self.latent_tokens.unsqueeze(0).expand(batch_size, -1, -1).to(x.dtype)
        latent_tokens = latent_tokens + encoder.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)
        x = encoder.ln_pre(x)
        return x.permute(1, 0, 2).contiguous()

    def _tail(self, y: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        batch_size = y.shape[1]
        latent_tokens = y.permute(1, 0, 2)[:, self.latent_token_start :]
        latent_tokens = encoder.ln_post(latent_tokens)
        if encoder.is_legacy:
            latent_tokens = latent_tokens.reshape(batch_size, encoder.width, encoder.num_latent_tokens, 1)
        else:
            latent_tokens = latent_tokens.reshape(batch_size, encoder.num_latent_tokens, encoder.width, 1).permute(0, 2, 1, 3)
        latent_tokens = encoder.conv_out(latent_tokens)
        return latent_tokens.reshape(batch_size, encoder.token_size, 1, encoder.num_latent_tokens).contiguous()

    def _chunk(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, self.channel_start : self.channel_start + self.channel_count].contiguous()

    def forward(self, pixel_values: torch.Tensor):
        prologue_out = self._prologue(pixel_values)
        if self.probe == "prologue_out":
            return self._chunk(prologue_out)

        block0_in = prologue_out
        block0_out = block0_in + self.block.attention_bhld_matmul(self.block.ln_1(block0_in))
        if self.block.mlp_ratio > 0:
            block0_out = block0_out + self.block.mlp(self.block.ln_2(block0_out))
        tail = self._tail(block0_out)
        # Return the boundary chunk first so simple board output-0 dumps capture it,
        # while tail keeps the block0 consumer live in the exported graph.
        return self._chunk(block0_in), tail


def first_output(value):  # noqa: ANN001
    if isinstance(value, tuple):
        return value[0]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--titok-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="yucornetto/tokenizer_titok_s128_imagenet")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--calibration-count", type=int, default=4)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--channel-start", type=int, default=0)
    parser.add_argument("--channel-count", type=int, default=4)
    parser.add_argument("--probes", nargs="*", default=list(PROBES), choices=PROBES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    output_dir = resolve_output_dir(REPO_ROOT, str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "host_generation_summary.json"
    manifest_path = resolve_input_path(str(args.manifest), REPO_ROOT)
    image_path = resolve_input_path(str(args.image), REPO_ROOT)
    calibration_images = load_manifest_records(manifest_path)[: args.calibration_count]

    report: dict[str, Any] = {
        "status": "started",
        "repo_id": args.repo_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "calibration_count": len(calibration_images),
        "calibration_images": [str(p) for p in calibration_images],
        "image": str(image_path),
        "channel_start": args.channel_start,
        "channel_count": args.channel_count,
        "logical_full_shape": [385, 1, 512],
        "probes": list(args.probes),
        "quantizer_backend": "ethosu",
        "quantization_profile": "int8",
    }

    try:
        print("[1/4] Loading TiTok")
        titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
        titok.requires_grad_(False)
        image_size = int(titok.config.dataset.preprocessing.crop_size)
        report["image_size"] = image_size

        image = load_image(image_path, image_size).to("cpu")
        example_input = load_image(calibration_images[0], image_size).to("cpu")
        input_blob = output_dir / "image_input_blob.bin"
        save_input_blob(input_blob, output_dir / "image_input_blob.json", image, str(image_path))
        report["image_input_blob"] = {"path": str(input_blob), "sha256": sha256_path(input_blob), "stats": tensor_stats(image)}

        probe_reports: dict[str, Any] = {}
        for probe in args.probes:
            print(f"[2/4] Processing {probe}")
            probe_dir = output_dir / probe
            probe_dir.mkdir(parents=True, exist_ok=True)
            module = BoundaryProbe(titok, probe, args.channel_start, args.channel_count).eval().to("cpu")
            module.requires_grad_(False)

            with torch.no_grad():
                float_output = first_output(module(image)).detach().to("cpu", dtype=torch.float32)
            float_path = probe_dir / "host_float_output.npz"
            save_float_npz(float_path, float_output)

            exported_program = export_encoder_program(module, example_input)
            prepared, compile_spec = prepare_exported_encoder_for_ptq(
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
            with torch.no_grad():
                for path in calibration_images:
                    prepared(load_image(path, image_size).to("cpu"))
            quantized = convert_encoder_after_ptq(prepared, backend="ethosu")
            with torch.no_grad():
                quantized_output = first_output(quantized(image)).detach().to("cpu", dtype=torch.float32)
            quantized_path = probe_dir / "host_quantized_pre_lowering_dequant_output.npz"
            save_float_npz(quantized_path, quantized_output)

            final_export = torch.export.export(quantized, (example_input,), strict=True)
            final_module = final_export.module()
            with torch.no_grad():
                final_output = first_output(final_module(image)).detach().to("cpu", dtype=torch.float32)
            final_path = probe_dir / "host_final_export_pre_lowering_dequant_output.npz"
            save_float_npz(final_path, final_output)

            int_record, all_quantize_records = capture_last_matching_quantize(final_module, image, list(final_output.shape))
            int_path = probe_dir / "host_final_export_pre_lowering_pre_dequant_int_output.npz"
            save_int_npz(int_path, int_record["value"])
            pre_qparams_path = probe_dir / "pre_lowering_qparams.csv"
            pre_qparam_rows = dump_qparams_csv(pre_qparams_path, final_module, "pre_lowering_final_export")

            print(f"[3/4] Lowering {probe}")
            partitioner = EthosUPartitioner(
                EthosUCompatCompileSpec(
                    "ethos-u65-256",
                    system_config="Ethos_U65_High_End",
                    memory_mode="Dedicated_Sram_384KB",
                    config_ini="Arm/vela.ini",
                    extra_flags=[],
                )
            )
            edge = to_edge_transform_and_lower(final_export, partitioner=[partitioner])
            edge = edge.transform([ReplaceSurvivingQdqWithOutVarPass()])
            post_graph = edge.exported_program().graph_module
            post_qparams_path = probe_dir / "post_lowering_qparams.csv"
            post_qparam_rows = dump_qparams_csv(post_qparams_path, post_graph, "post_lowering_qdq_fix")
            executorch_program = edge.to_executorch()
            pte_path = probe_dir / f"{probe}.pte"
            pte_path.write_bytes(executorch_program.buffer)

            probe_report = {
                "probe": probe,
                "chunk": {
                    "channel_start": args.channel_start,
                    "channel_count": args.channel_count,
                    "logical_source_shape": [385, 1, 512],
                    "output_shape": list(final_output.shape),
                },
                "float_output": {"path": str(float_path), "stats": tensor_stats(float_output)},
                "host_quantized_pre_lowering_dequant_output": {"path": str(quantized_path), "stats": tensor_stats(quantized_output)},
                "host_final_export_pre_lowering_dequant_output": {"path": str(final_path), "stats": tensor_stats(final_output)},
                "host_final_export_pre_lowering_pre_dequant_int_output": {
                    "path": str(int_path),
                    "node": {k: v for k, v in int_record.items() if k != "value"},
                    "stats": tensor_stats(int_record["value"]),
                },
                "captured_quantize_nodes": [{k: v for k, v in record.items() if k != "value"} for record in all_quantize_records],
                "pre_lowering_qparams_csv": {"path": str(pre_qparams_path), "row_count": len(pre_qparam_rows)},
                "post_lowering_qparams_csv": {"path": str(post_qparams_path), "row_count": len(post_qparam_rows)},
                "final_export_graph_summary": summarize_fx_graph(final_export, "final_export"),
                "post_partition_qdq_fix_graph_summary": summarize_fx_graph(post_graph, "post_partition_qdq_fix"),
                "runtime_program_summary": summarize_executorch_program(executorch_program),
                "pte_path": str(pte_path),
                "pte_size": pte_path.stat().st_size,
                "pte_sha256": sha256_path(pte_path),
                "compile_spec_flags": compile_spec.compiler_flags if compile_spec is not None else None,
                "host_comparisons": {
                    "quantized_pre_lowering_vs_final_export_pre_lowering_dequant": compare_arrays(
                        quantized_output.numpy(),
                        final_output.numpy(),
                    )
                },
            }
            (probe_dir / "host_generation_summary.json").write_text(json.dumps(probe_report, indent=2) + "\n")
            probe_reports[probe] = probe_report

        report["probe_reports"] = probe_reports
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

    print("[4/4] Done")
    print(json.dumps({"status": report["status"], "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
