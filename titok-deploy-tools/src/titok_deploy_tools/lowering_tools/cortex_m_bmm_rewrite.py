from pathlib import Path
import importlib
import sys
from typing import Any


def _prefer_local_executorch_checkout() -> Path | None:
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        candidate = parent / "executorch-main" / "src"
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str in sys.path:
                sys.path.remove(candidate_str)
            sys.path.insert(0, candidate_str)
            return candidate
    return None


_prefer_local_executorch_checkout()

import executorch.backends.cortex_m.ops.operators  # noqa: F401
import torch
from executorch.backends.cortex_m.passes.passes_utils import quantize_multiplier_aot
from executorch.backends.transforms.utils import get_param_tensor, is_param_node
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.exir.pass_base import PassResult
from executorch.exir.program._program import _transform


def _as_scalar(exported_program, value: Any) -> float | int:
    if hasattr(value, "name") and is_param_node(exported_program, value):
        value = get_param_tensor(exported_program, value)
    if hasattr(value, "numel") and value.numel() == 1:
        value = value.item()
    return value


def _target_name(node: Any) -> str:
    return str(getattr(node, "target", ""))


def _shape_list(node: Any) -> list[int] | None:
    meta_val = getattr(node, "meta", {}).get("val")
    shape = getattr(meta_val, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except Exception:
        return None


def _get_cortex_m_op(name: str):
    importlib.import_module("executorch.backends.cortex_m.ops.operators")
    return getattr(torch.ops.cortex_m, name).default


def _is_qualified_int8_qdq(exported_program, node: Any) -> bool:
    args = getattr(node, "args", ())
    if len(args) < 6:
        return False
    try:
        qmin = int(_as_scalar(exported_program, args[3]))
        qmax = int(_as_scalar(exported_program, args[4]))
    except Exception:
        return False
    dtype = args[5]
    return (
        qmin >= torch.iinfo(torch.int8).min
        and qmax <= torch.iinfo(torch.int8).max
        and dtype == torch.int8
    )


class RewriteQdqBmmToCortexMPass(XNNPACKPass):
    """Rewrite dq -> bmm -> q islands into cortex_m quantized_batch_matmul.

    This pass is intentionally narrow: it only touches explicit quantized BMM
    fallback islands and leaves the rest of the graph unchanged so Ethos-U
    partitioning can still claim the surrounding ops.
    """

    def call(self, graph_module):
        modified = False
        graph = graph_module.graph

        for node in list(graph.nodes):
            if node.op != "call_function" or len(node.args) != 2:
                continue

            lhs_dq, rhs_dq = node.args
            if (
                _target_name(lhs_dq)
                != "quantized_decomposed.dequantize_per_tensor.default"
                or _target_name(rhs_dq)
                != "quantized_decomposed.dequantize_per_tensor.default"
            ):
                continue

            lhs_shape = _shape_list(lhs_dq)
            rhs_shape = _shape_list(rhs_dq)
            out_shape = _shape_list(node)
            if lhs_shape is None or rhs_shape is None or out_shape is None:
                continue
            if len(lhs_shape) != 3 or len(rhs_shape) != 3 or len(out_shape) != 3:
                continue
            if lhs_shape[0] != rhs_shape[0] or out_shape[0] != lhs_shape[0]:
                continue
            if lhs_shape[2] != rhs_shape[1]:
                continue
            if out_shape[1] != lhs_shape[1] or out_shape[2] != rhs_shape[2]:
                continue

            users = list(node.users)
            if len(users) != 1:
                continue
            q_node = users[0]
            if (
                q_node.op != "call_function"
                or _target_name(q_node)
                != "quantized_decomposed.quantize_per_tensor.default"
                or q_node.args[0] is not node
            ):
                continue

            lhs_q = lhs_dq.args[0]
            rhs_q = rhs_dq.args[0]
            lhs_scale = float(_as_scalar(self.exported_program, lhs_dq.args[1]))
            lhs_zp = int(_as_scalar(self.exported_program, lhs_dq.args[2]))
            rhs_scale = float(_as_scalar(self.exported_program, rhs_dq.args[1]))
            rhs_zp = int(_as_scalar(self.exported_program, rhs_dq.args[2]))
            output_scale = float(_as_scalar(self.exported_program, q_node.args[1]))
            output_zp = int(_as_scalar(self.exported_program, q_node.args[2]))
            output_mult, output_shift = quantize_multiplier_aot(
                (lhs_scale * rhs_scale) / output_scale
            )

            with graph.inserting_before(q_node):
                rhs_transposed = graph.create_node(
                    "call_function",
                    target=_get_cortex_m_op("transpose"),
                    args=(rhs_q, [0, 2, 1]),
                    kwargs={},
                )
                rhs_transposed.meta = dict(getattr(rhs_q, "meta", {}))

                cortex_m_bmm = graph.create_node(
                    "call_function",
                    target=_get_cortex_m_op("quantized_batch_matmul"),
                    args=(
                        lhs_q,
                        -lhs_zp,
                        rhs_transposed,
                        -rhs_zp,
                        output_zp,
                        output_mult,
                        output_shift,
                    ),
                    kwargs={},
                )
                cortex_m_bmm.meta = dict(q_node.meta)

            q_node.replace_all_uses_with(cortex_m_bmm)
            graph.erase_node(q_node)
            if len(node.users) == 0:
                graph.erase_node(node)
            modified = True

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue

            target_name = _target_name(node)
            if target_name == "quantized_decomposed.quantize_per_tensor.default":
                if not _is_qualified_int8_qdq(self.exported_program, node):
                    continue
                node.target = _get_cortex_m_op("quantize_per_tensor")
                modified = True
            elif target_name == "quantized_decomposed.dequantize_per_tensor.default":
                node.target = _get_cortex_m_op("dequantize_per_tensor")
                modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)


def rewrite_qdq_bmm_to_cortex_m(exported_program):
    return _transform(exported_program, RewriteQdqBmmToCortexMPass(exported_program))
