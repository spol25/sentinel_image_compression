from pathlib import Path
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

import executorch.kernels.quantized  # noqa: F401
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes._quant_patterns_and_replacements import (  # noqa: F401
    quantized_decomposed_lib,
)
from executorch.exir.pass_base import PassResult
from torch.fx.passes.infra.pass_base import PassBase


_QDQ_OUT_SCHEMA_LIB: torch.library.Library | None = None


def _ensure_quantized_decomposed_out_overloads() -> None:
    """Register q/dq out schemas when this ExecuTorch build only exposes default.

    Some ExecuTorch checkouts have C++ schemas and kernels for these out variants
    but do not install matching Python torch.library schemas. The post-partition
    rewrite needs the Python handles so the final to_executorch() pass can
    serialize planned outputs instead of leaving functional Q/DQ nodes behind.
    """

    missing = []
    for name in ("quantize_per_tensor", "dequantize_per_tensor"):
        op = getattr(torch.ops.quantized_decomposed, name)
        if "out" not in op.overloads():
            missing.append(name)

    if not missing:
        return

    global _QDQ_OUT_SCHEMA_LIB
    lib = torch.library.Library("quantized_decomposed", "FRAGMENT")
    if "quantize_per_tensor" in missing:
        lib.define(
            "quantize_per_tensor.out(Tensor input, float scale, int zero_point, "
            "int quant_min, int quant_max, ScalarType dtype, *, "
            "Tensor(a!) out) -> Tensor(a!)"
        )
    if "dequantize_per_tensor" in missing:
        lib.define(
            "dequantize_per_tensor.out(Tensor input, float scale, int zero_point, "
            "int quant_min, int quant_max, ScalarType dtype, *, "
            "ScalarType? out_dtype=None, Tensor(a!) out) -> Tensor(a!)"
        )
    _QDQ_OUT_SCHEMA_LIB = lib


_ensure_quantized_decomposed_out_overloads()


def _as_scalar(value: Any) -> float | int:
    if hasattr(value, "numel") and value.numel() == 1:
        value = value.item()
    return value


def _target_name(node: Any) -> str:
    target = getattr(node, "target", None)
    schema = getattr(target, "_schema", None)
    if schema is not None:
        overload = getattr(schema, "overload_name", "")
        return f"{schema.name}.{overload}" if overload else schema.name
    return str(target)


def _matches_target(node: Any, qualified_name: str) -> bool:
    target_name = _target_name(node)
    return target_name == qualified_name or target_name == f"{qualified_name}.default"


def _is_qualified_int8_quantize(node: Any) -> bool:
    args = getattr(node, "args", ())
    if len(args) < 6:
        return False
    try:
        qmin = int(_as_scalar(args[3]))
        qmax = int(_as_scalar(args[4]))
    except Exception:
        return False
    dtype = args[5]
    return (
        qmin >= torch.iinfo(torch.int8).min
        and qmax <= torch.iinfo(torch.int8).max
        and dtype == torch.int8
    )


class ReplaceSurvivingQdqWithOutVarPass(PassBase):
    """Make post-partition leftover q/dq nodes convertible to .out overloads.

    This pass is meant to run after Ethos-U partitioning has already happened on
    an EdgeProgramManager. The actual functional-to-out rewrite must be done by
    ExecuTorch's ToOutVarPass because it also adds the planned output argument.
    This pass therefore only exists to ensure the local Python op registry has
    q/dq out schemas before to_executorch() runs.
    """

    def call(self, graph_module):
        _ensure_quantized_decomposed_out_overloads()
        return PassResult(graph_module, False)


# Backward-compatible alias for older script imports while we migrate away from
# the cortex_m-targeted version of this post-partition rewrite.
ReplaceSurvivingQdqWithCortexMPass = ReplaceSurvivingQdqWithOutVarPass
