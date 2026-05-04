import hashlib
from collections import Counter
from typing import Any


def _safe_shape(meta_val):
    if meta_val is None:
        return None
    shape = getattr(meta_val, "shape", None)
    if shape is None:
        return None
    try:
        return list(shape)
    except TypeError:
        return str(shape)


def summarize_fx_graph(graph_like: Any, stage_name: str) -> dict[str, Any]:
    """Return a JSON-friendly summary of an FX-style graph boundary."""

    if hasattr(graph_like, "graph_module"):
        graph_module = graph_like.graph_module
    else:
        graph_module = graph_like
    graph = graph_module.graph

    op_counts: Counter[str] = Counter()
    op_type_counts: Counter[str] = Counter()
    nodes: list[dict[str, Any]] = []
    node_signature: list[str] = []

    for node in graph.nodes:
        target = str(node.target)
        op_counts[target] += 1
        op_type_counts[node.op] += 1
        node_signature.append(f"{node.op}|{target}")
        tensor_meta = node.meta.get("val") if isinstance(node.meta, dict) else None
        nodes.append(
            {
                "name": node.name,
                "op": node.op,
                "target": target,
                "shape": _safe_shape(tensor_meta),
            }
        )

    return {
        "stage": stage_name,
        "boundary_type": "fx_graph",
        "node_count": len(nodes),
        "unique_op_count": len(op_counts),
        "op_type_counts": dict(sorted(op_type_counts.items())),
        "target_counts": dict(sorted(op_counts.items())),
        "graph_signature": hashlib.sha256("\n".join(node_signature).encode("utf-8")).hexdigest(),
        "nodes": nodes,
    }
