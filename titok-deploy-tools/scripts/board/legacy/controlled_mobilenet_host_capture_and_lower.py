#!/usr/bin/env python3
"""Controlled MobileNetV2 host capture and Ethos-U lowering."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
from executorch.exir import to_edge_transform_and_lower
from PIL import Image
from torch.export import export, export_for_training
from torch.fx import Interpreter
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from titok_deploy_tools.lowering_tools.ethosu_compat import EthosUCompatCompileSpec
from titok_deploy_tools.lowering_tools.executorch_summary import summarize_executorch_program
from titok_deploy_tools.lowering_tools.graph_summary import summarize_fx_graph
from titok_deploy_tools.lowering_tools.post_partition_qdq_fix import ReplaceSurvivingQdqWithOutVarPass
from titok_deploy_tools.ptq_tools.ptq import load_manifest_records


INPUT_MAGIC = b"ETINP001"
INPUT_HEADER_SIZE = 64
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def load_mobilenet_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((224, 224), Image.Resampling.BICUBIC)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
    image_np = np.ascontiguousarray(np.transpose(image_np, (2, 0, 1))[None, ...])
    return torch.from_numpy(image_np)


def save_input_blob(path: Path, metadata_path: Path, tensor: torch.Tensor, image_path: Path) -> None:
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
    metadata = {
        "image": str(image_path),
        "output": str(path),
        "magic": INPUT_MAGIC.decode("ascii"),
        "header_size": INPUT_HEADER_SIZE,
        "payload_size": len(payload),
        "dtype": "float32",
        "shape": list(array.shape),
        "preprocessing": "resize 224x224 bicubic, RGB, /255, ImageNet mean/std, NCHW",
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "sha256": sha256_path(path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def save_float_npz(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, output=tensor.detach().to("cpu", dtype=torch.float32).contiguous().numpy())


def save_int_npz(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().to("cpu").contiguous()
    np.savez(path, output_int=value.numpy())


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
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--calibration-count", type=int, default=4)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--artifact-name", default="mobilenetv2_controlled_u65_dedicated_sram_384kb.pte")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "mobilenet_controlled_host_capture_and_lower_summary.json"

    calibration_images = load_manifest_records(args.manifest)[: args.calibration_count]
    if not calibration_images:
        raise SystemExit("Calibration image list is empty.")

    report: dict[str, Any] = {
        "status": "started",
        "model": "torchvision.models.mobilenet_v2",
        "weights": "MobileNet_V2_Weights.IMAGENET1K_V1",
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_path(args.manifest),
        "calibration_count": len(calibration_images),
        "calibration_images": [str(p) for p in calibration_images],
        "image": str(args.image),
        "input_shape": [1, 3, 224, 224],
        "preprocessing": "resize 224x224 bicubic, RGB, /255, ImageNet mean/std, NCHW",
        "quantizer_backend": "ethosu",
        "quantization_profile": "torchao EthosUQuantizer global symmetric",
        "ethos_target": "ethos-u65-256",
        "ethos_system_config": "Ethos_U65_High_End",
        "ethos_memory_mode": "Dedicated_Sram_384KB",
    }

    try:
        print("[1/8] Loading MobileNetV2")
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).eval().to("cpu")
        model.requires_grad_(False)

        image = load_mobilenet_tensor(args.image)
        example_input = load_mobilenet_tensor(calibration_images[0])
        input_blob_path = output_dir / "mobilenet_controlled_input_blob.bin"
        input_blob_json_path = output_dir / "mobilenet_controlled_input_blob.json"
        save_input_blob(input_blob_path, input_blob_json_path, image, args.image)
        report["input_blob"] = {
            "path": str(input_blob_path),
            "metadata": str(input_blob_json_path),
            "sha256": sha256_path(input_blob_path),
            "stats": tensor_stats(image),
        }

        print("[2/8] Running float host model")
        with torch.no_grad():
            float_output = model(image).detach().to("cpu", dtype=torch.float32)
        float_path = output_dir / "host_float_mobilenet_output.npz"
        save_float_npz(float_path, float_output)
        report["host_float_output"] = {"path": str(float_path), "stats": tensor_stats(float_output)}

        print("[3/8] Exporting and preparing PTQ")
        exported_program = export_for_training(model, (example_input,)).module(check_guards=False)
        compile_spec = EthosUCompatCompileSpec(
            target="ethos-u65-256",
            system_config="Ethos_U65_High_End",
            memory_mode="Dedicated_Sram_384KB",
        )
        vela_dir = output_dir / "vela_intermediates"
        compile_spec.dump_intermediate_artifacts_to(str(vela_dir))
        report["vela_intermediates_dir"] = str(vela_dir)
        quantizer = EthosUQuantizer(compile_spec)
        quantizer.set_global(get_symmetric_quantization_config())
        prepared = prepare_pt2e(exported_program, quantizer)

        print(f"[4/8] Calibrating on {len(calibration_images)} fixed image(s)")
        with torch.no_grad():
            for calibration_image in calibration_images:
                prepared(load_mobilenet_tensor(calibration_image))

        print("[5/8] Converting quantized model")
        quantized = convert_pt2e(prepared)
        with torch.no_grad():
            quantized_output = quantized(image).detach().to("cpu", dtype=torch.float32)
        quantized_float_path = output_dir / "host_quantized_pre_lowering_dequant_output.npz"
        save_float_npz(quantized_float_path, quantized_output)
        report["host_quantized_pre_lowering_dequant_output"] = {
            "path": str(quantized_float_path),
            "stats": tensor_stats(quantized_output),
        }

        print("[6/8] Exporting quantized graph and capturing final int output")
        final_export = export(quantized, (example_input,), strict=True)
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
        report["host_final_export_pre_lowering_dequant_output"] = {
            "path": str(final_export_float_path),
            "stats": tensor_stats(final_export_output),
        }
        report["host_final_export_pre_lowering_pre_dequant_int_output"] = {
            "path": str(int_path),
            "node": {k: v for k, v in int_record.items() if k != "value"},
            "stats": tensor_stats(int_tensor),
        }
        report["captured_quantize_nodes"] = [
            {k: v for k, v in record.items() if k != "value"} for record in all_quantize_records
        ]
        report["final_export_graph_summary"] = summarize_fx_graph(final_export, "final_export")

        print("[7/8] Lowering same exported quantized graph to Ethos-U")
        partitioner = EthosUPartitioner(compile_spec)
        edge = to_edge_transform_and_lower(final_export, partitioner=[partitioner])
        report["post_partition_graph_summary"] = summarize_fx_graph(
            edge.exported_program().graph_module,
            "post_partition",
        )
        edge = edge.transform([ReplaceSurvivingQdqWithOutVarPass()])
        report["post_partition_qdq_fix_graph_summary"] = summarize_fx_graph(
            edge.exported_program().graph_module,
            "post_partition_qdq_fix",
        )

        print("[8/8] Serializing PTE")
        executorch_program = edge.to_executorch()
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
