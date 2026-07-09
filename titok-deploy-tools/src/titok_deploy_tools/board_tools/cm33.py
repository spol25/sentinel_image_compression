from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

from titok_deploy_tools.board_tools.artifacts import sha256_path


INPUT_MAGIC = b"ETINP001"
OUTPUT_MAGIC = b"ETOUT001"
HEADER_SIZE = 64


def tensor_to_input_blob_payload(tensor: torch.Tensor | np.ndarray) -> tuple[bytes, dict[str, Any]]:
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
    else:
        array = np.ascontiguousarray(tensor, dtype=np.float32)
    array = array.astype("<f4", copy=False)
    payload = array.tobytes(order="C")
    header = bytearray(HEADER_SIZE)
    header[: len(INPUT_MAGIC)] = INPUT_MAGIC
    struct.pack_into("<I", header, 8, len(payload))
    struct.pack_into("<I", header, 12, array.ndim)
    for i, dim in enumerate(array.shape[:6]):
        struct.pack_into("<I", header, 16 + 4 * i, int(dim))
    metadata = {
        "magic": INPUT_MAGIC.decode("ascii"),
        "header_size": HEADER_SIZE,
        "payload_size": len(payload),
        "dtype": "float32",
        "shape": list(array.shape),
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
        "mean": float(array.mean()) if array.size else None,
    }
    return bytes(header) + payload, metadata


def write_input_blob(path: Path, tensor: torch.Tensor | np.ndarray, *, metadata_path: Path | None = None, source: str | None = None) -> dict[str, Any]:
    blob, metadata = tensor_to_input_blob_payload(tensor)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    metadata.update({"output": str(path), "sha256": sha256_path(path)})
    if source is not None:
        metadata["source"] = source
    (metadata_path or path.with_suffix(".json")).write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def parse_output_blob(blob_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    data = blob_path.read_bytes()
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Blob is too small: {len(data)} bytes")
    if data[: len(OUTPUT_MAGIC)] != OUTPUT_MAGIC:
        raise ValueError(f"Bad magic {data[:len(OUTPUT_MAGIC)]!r}; expected {OUTPUT_MAGIC!r}")

    header_size = struct.unpack_from("<I", data, 8)[0]
    num_outputs = struct.unpack_from("<I", data, 12)[0]
    invoke_status = struct.unpack_from("<I", data, 16)[0]
    get_outputs_status = struct.unpack_from("<I", data, 20)[0]
    dtype_id = struct.unpack_from("<I", data, 24)[0]
    dim = struct.unpack_from("<I", data, 28)[0]
    nbytes = struct.unpack_from("<I", data, 32)[0]
    shape = [struct.unpack_from("<I", data, 36 + 4 * i)[0] for i in range(min(dim, 6))]
    payload = data[header_size : header_size + nbytes]

    dtype = np.float32 if nbytes == int(np.prod(shape or [0])) * 4 else np.uint8
    output = np.frombuffer(payload, dtype=dtype).copy()
    if shape and output.size == int(np.prod(shape)):
        output = output.reshape(shape)

    metadata = {
        "blob": str(blob_path),
        "magic": OUTPUT_MAGIC.decode("ascii"),
        "header_size": header_size,
        "num_outputs": num_outputs,
        "invoke_status": invoke_status,
        "get_outputs_status": get_outputs_status,
        "dtype_id": dtype_id,
        "dtype_inferred": str(output.dtype),
        "shape": shape,
        "nbytes": nbytes,
        "min": float(output.min()) if output.size else None,
        "max": float(output.max()) if output.size else None,
        "mean": float(output.mean()) if output.size else None,
    }
    return output, metadata
