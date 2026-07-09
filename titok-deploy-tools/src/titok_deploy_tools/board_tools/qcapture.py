from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.fx import Interpreter


QPARAM_FIELDNAMES = [
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


def capture_last_matching_quantize(module: torch.fx.GraphModule, tensor: torch.Tensor, output_shape: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    interp = QuantizeCaptureInterpreter(module)
    _ = interp.run(tensor)
    matching = [
        record
        for record in interp.quantize_records
        if record["shape"] == output_shape and isinstance(record["value"], torch.Tensor)
    ]
    if not matching:
        raise RuntimeError(f"No quantize_per_tensor node found with output shape {output_shape}")
    return matching[-1], interp.quantize_records


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
    return {"value": str(value), "shape": "", "min": "", "max": "", "first_8": ""}


def dump_qparams_csv(path: Path, module: torch.fx.GraphModule, stage: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for node in module.graph.nodes:
        target = str(node.target)
        if not any(
            token in target
            for token in ("quantize_per_tensor", "dequantize_per_tensor", "quantize_per_channel", "dequantize_per_channel")
        ):
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

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QPARAM_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return rows
