import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import torch
from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.ethosu import EthosUPartitioner
from executorch.exir import to_edge_transform_and_lower
from tosa.Op import Op
from tosa.TosaGraph import TosaGraph
from tosa.TransposeAttribute import TransposeAttribute

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.ethosu_compat import EthosUCompatCompileSpec
from titok_deploy_tools.ptq import (
    build_encoder_quantizer_split,
    calibrate_prepared_encoder,
    convert_encoder_after_ptq,
    export_encoder_program,
    load_manifest_records,
    prepare_exported_encoder_for_ptq,
)
from titok_deploy_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.utils import load_image, resolve_input_path, resolve_output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect transpose/permute patterns in the Arm Ethos-U quantized TiTok-S-128 encoder."
    )
    parser.add_argument("--titok-root", required=True, help="Path to a separate 1d-tokenizer checkout.")
    parser.add_argument("--manifest", required=True, help="Calibration manifest used for the quick PTQ prep.")
    parser.add_argument("--repo-id", default="yucornetto/tokenizer_titok_s128_imagenet")
    parser.add_argument("--calibration-count", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="scripts/export_and_lower/inspect_and_debug/generated/arm_lowering_inspect",
    )
    parser.add_argument("--summary-name", default="ethosu_u65_transpose_inspection.json")
    parser.add_argument("--ethos-target", default="ethos-u65-256")
    parser.add_argument("--ethos-config-ini", default="Arm/vela.ini")
    parser.add_argument("--per-channel", action="store_true")
    return parser.parse_args()


def _fake_tensor_shape_dtype(node):
    val = node.meta.get("val")
    shape = list(val.shape) if hasattr(val, "shape") else None
    dtype = str(val.dtype) if hasattr(val, "dtype") else None
    return shape, dtype


def _collect_export_transposes(graph_module):
    items = []
    for index, node in enumerate(graph_module.graph.nodes):
        target = str(node.target)
        if not any(k in target for k in ["permute", "transpose", "_to_dim_order_copy", "dim_order"]):
            continue
        shape, dtype = _fake_tensor_shape_dtype(node)
        items.append(
            {
                "index": index,
                "name": node.name,
                "op": node.op,
                "target": target,
                "shape": shape,
                "dtype": dtype,
                "args": str(node.args),
                "users": [user.name for user in node.users],
            }
        )
    return items


def _collect_tosa_transposes(tosa_path: Path):
    buf = tosa_path.read_bytes()
    graph = TosaGraph.GetRootAsTosaGraph(buf, 0)
    region = graph.Regions(0)
    block = region.Blocks(0)

    tensors = {}
    for i in range(block.TensorsLength()):
        tensor = block.Tensors(i)
        tensors[tensor.Name().decode()] = [tensor.Shape(j) for j in range(tensor.ShapeLength())]

    items = []
    for i in range(block.OperatorsLength()):
        op = block.Operators(i)
        if op.Op() != Op.TRANSPOSE:
            continue
        inputs = [op.Inputs(j).decode() for j in range(op.InputsLength())]
        outputs = [op.Outputs(j).decode() for j in range(op.OutputsLength())]
        attr = op.Attribute()
        transpose_attr = TransposeAttribute()
        transpose_attr.Init(attr.Bytes, attr.Pos)
        perms = [transpose_attr.Perms(j) for j in range(transpose_attr.PermsLength())]
        items.append(
            {
                "index": i,
                "inputs": inputs,
                "input_shapes": [tensors.get(name) for name in inputs],
                "outputs": outputs,
                "output_shapes": [tensors.get(name) for name in outputs],
                "perms": perms,
            }
        )
    return items


def main():
    args = parse_args()
    add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    manifest_path = resolve_input_path(args.manifest, REPO_ROOT)
    image_paths = load_manifest_records(manifest_path)[: args.calibration_count]
    if not image_paths:
        raise SystemExit("Calibration manifest is empty.")

    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
    summary_path = output_dir / Path(args.summary_name).name
    tosa_dir = output_dir / "tosa_debug"
    tosa_dir.mkdir(parents=True, exist_ok=True)

    titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
    titok.requires_grad_(False)
    encoder_only, _, _ = build_encoder_quantizer_split(titok)
    encoder_only = encoder_only.eval().to("cpu")
    encoder_only.requires_grad_(False)

    image_size = int(titok.config.dataset.preprocessing.crop_size)
    example_input = load_image(image_paths[0], image_size).to("cpu")

    exported_program = export_encoder_program(encoder_only, example_input)
    prepared_encoder, _ = prepare_exported_encoder_for_ptq(
        exported_program,
        backend="ethosu",
        is_per_channel=args.per_channel,
        ethos_target=args.ethos_target,
        ethos_config_ini=args.ethos_config_ini,
    )
    calibrate_prepared_encoder(prepared_encoder, image_paths, image_size)
    quantized_encoder = convert_encoder_after_ptq(prepared_encoder, backend="ethosu")
    final_export = torch.export.export(quantized_encoder, (example_input,), strict=True)

    export_transposes = _collect_export_transposes(final_export.graph_module)

    compile_spec = (
        EthosUCompatCompileSpec(args.ethos_target, config_ini=args.ethos_config_ini)
        .dump_intermediate_artifacts_to(str(tosa_dir))
        .dump_debug_info(ArmCompileSpec.DebugMode.TOSA)
    )
    partitioner = EthosUPartitioner(compile_spec)

    lowering_error = None
    try:
        to_edge_transform_and_lower(final_export, partitioner=[partitioner])
    except Exception as exc:
        lowering_error = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    tosa_files = sorted(tosa_dir.glob("*.tosa"))
    tosa_transposes = _collect_tosa_transposes(tosa_files[0]) if tosa_files else []

    export_target_counts = Counter(item["target"] for item in export_transposes)
    tosa_perm_counts = Counter(tuple(item["perms"]) for item in tosa_transposes)

    summary = {
        "repo_id": args.repo_id,
        "manifest_path": str(manifest_path),
        "num_calibration_images": len(image_paths),
        "ethos_target": args.ethos_target,
        "per_channel": args.per_channel,
        "lowering_error": lowering_error,
        "export_transpose_count": len(export_transposes),
        "export_target_counts": dict(export_target_counts),
        "export_transposes": export_transposes,
        "tosa_file": str(tosa_files[0]) if tosa_files else None,
        "tosa_transpose_count": len(tosa_transposes),
        "tosa_permutation_counts": {str(list(k)): v for k, v in sorted(tosa_perm_counts.items())},
        "tosa_transposes": tosa_transposes,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved transpose inspection summary to {summary_path}")


if __name__ == "__main__":
    main()
