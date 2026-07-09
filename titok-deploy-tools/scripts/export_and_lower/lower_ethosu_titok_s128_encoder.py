import argparse
import json
from pathlib import Path
import sys
import traceback

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

from titok_deploy_tools.lowering_tools.cortex_m_bmm_rewrite import rewrite_qdq_bmm_to_cortex_m
from titok_deploy_tools.lowering_tools.ethosu_compat import EthosUCompatCompileSpec
from titok_deploy_tools.lowering_tools.executorch_summary import summarize_executorch_program
from titok_deploy_tools.lowering_tools.graph_summary import summarize_fx_graph
from titok_deploy_tools.lowering_tools.post_partition_qdq_fix import ReplaceSurvivingQdqWithOutVarPass
from titok_deploy_tools.ptq_tools.ptq import (
    build_encoder_quantizer_split,
    calibrate_prepared_encoder,
    convert_encoder_after_ptq,
    export_encoder_program,
    load_manifest_records,
    prepare_exported_encoder_for_ptq,
)
from titok_deploy_tools.wrapper_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.wrapper_tools.utils import load_image, resolve_input_path, resolve_output_dir


def tensor_stats(tensor: torch.Tensor) -> dict:
    value = tensor.detach().to("cpu", dtype=torch.float32)
    flat = value.reshape(-1)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "first_8": [float(x) for x in flat[:8].tolist()],
    }


def save_npz_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().to("cpu", dtype=torch.float32).contiguous()
    np.savez(path, output=value.numpy())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize the TiTok-S-128 encoder with the Arm Ethos-U backend and attempt U65 lowering."
    )
    parser.add_argument("--titok-root", required=True, help="Path to a separate 1d-tokenizer checkout.")
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_s128_imagenet",
        help="Hugging Face repo for the pretrained TiTok-S-128 tokenizer.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Calibration manifest used for the encoder PTQ preparation.",
    )
    parser.add_argument(
        "--calibration-count",
        type=int,
        default=4,
        help="How many manifest images to use for a quick lowering smoke test.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/arm_lowering",
        help="Directory where the lowering summary will be written.",
    )
    parser.add_argument(
        "--encoder-variant",
        choices=(
            "baseline",
            "einsum_attention",
            "reshape_batch",
            "bmm_attention",
            "source_matmul_attention",
            "source_sdpa_attention",
        ),
        default="baseline",
        help=(
            "Encoder wrapper to lower. baseline uses the TiTok fork default "
            "attention implementation."
        ),
    )
    parser.add_argument(
        "--prefix-num-blocks",
        type=int,
        default=None,
        help=(
            "If set, lower only the encoder prefix through this many transformer "
            "blocks. Currently supported for baseline and source_sdpa_attention."
        ),
    )
    parser.add_argument(
        "--summary-name",
        default="ethosu_u65_lowering_summary.json",
        help="Filename for the lowering summary JSON.",
    )
    parser.add_argument(
        "--artifact-name",
        default="titok_s128_encoder_ethosu_u65_a16w8.pte",
        help="Filename for the lowered ExecuTorch .pte artifact.",
    )
    parser.add_argument(
        "--ethos-target",
        default="ethos-u65-256",
        help="Ethos-U accelerator target string.",
    )
    parser.add_argument(
        "--ethos-system-config",
        default=None,
        help="Optional Vela system_config override.",
    )
    parser.add_argument(
        "--ethos-memory-mode",
        default=None,
        help="Optional Vela memory_mode override.",
    )
    parser.add_argument(
        "--ethos-config-ini",
        default="Arm/vela.ini",
        help="Path to the Vela .ini file used in the compile spec.",
    )
    parser.add_argument(
        "--ethos-extra-flag",
        action="append",
        default=[],
        help="Additional compiler flag to append to the Ethos-U compile spec. May be repeated.",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Use per-channel symmetric quantization for supported weights.",
    )
    parser.add_argument(
        "--quantization-profile",
        choices=("int8", "a16w8"),
        default="a16w8",
        help="Quantization profile to use for the encoder PTQ flow.",
    )
    parser.add_argument(
        "--strict-failure",
        action="store_true",
        help="Exit non-zero if lowering fails after writing the summary.",
    )
    parser.add_argument(
        "--post-partition-qdq-out-fix",
        "--post-partition-cortexm-qdq-fix",
        dest="post_partition_qdq_out_fix",
        action="store_true",
        help="After Ethos-U partitioning, rewrite only the surviving quantized_decomposed q/dq ops to explicit .out overloads.",
    )
    parser.add_argument(
        "--quantize-matmul",
        action="store_true",
        help="Ask the Ethos-U quantizer to quantize aten.matmul nodes when supported.",
    )
    parser.add_argument(
        "--rewrite-cortexm-bmm",
        action="store_true",
        help=(
            "Rewrite quantized BMM fallback islands to cortex_m::quantized_batch_matmul "
            "before Ethos-U partitioning."
        ),
    )
    parser.add_argument(
        "--capture-image",
        default=None,
        help=(
            "Optional image to run through this invocation's quantized encoder "
            "and exported pre-partition graph."
        ),
    )
    parser.add_argument(
        "--capture-output-prefix",
        default="pre_lowering_capture",
        help="Filename prefix for optional --capture-image artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    manifest_path = resolve_input_path(args.manifest, REPO_ROOT)
    image_paths = load_manifest_records(manifest_path)
    if not image_paths:
        raise SystemExit("Calibration manifest is empty.")
    image_paths = image_paths[: args.calibration_count]

    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
    summary_path = output_dir / Path(args.summary_name).name
    artifact_path = output_dir / Path(args.artifact_name).name
    capture_image_path = (
        resolve_input_path(args.capture_image, REPO_ROOT)
        if args.capture_image is not None
        else None
    )

    summary = {
        "repo_id": args.repo_id,
        "manifest_path": str(manifest_path),
        "num_calibration_images": len(image_paths),
        "quantizer_backend": "ethosu",
        "encoder_variant": args.encoder_variant,
        "prefix_num_blocks": args.prefix_num_blocks,
        "quantize_matmul": args.quantize_matmul,
        "fallback_rewrite": (
            "qdq_bmm_to_cortex_m_quantized_batch_matmul"
            if args.rewrite_cortexm_bmm
            else None
        ),
        "post_partition_qdq_out_fix": args.post_partition_qdq_out_fix,
        "ethos_target": args.ethos_target,
        "ethos_system_config": args.ethos_system_config,
        "ethos_memory_mode": args.ethos_memory_mode,
        "ethos_config_ini": args.ethos_config_ini,
        "ethos_extra_flags": args.ethos_extra_flag,
        "per_channel": args.per_channel,
        "quantization_profile": args.quantization_profile,
        "status": "started",
    }

    try:
        print("[1/7] Loading TiTok model on CPU")
        titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
        titok.requires_grad_(False)

        print(f"[2/8] Building {args.encoder_variant} encoder wrapper")
        encoder_only, _, _ = build_encoder_quantizer_split(
            titok,
            encoder_variant=args.encoder_variant,
            prefix_num_blocks=args.prefix_num_blocks,
        )
        encoder_only = encoder_only.eval().to("cpu")
        encoder_only.requires_grad_(False)
        summary["wrapper_variant"] = encoder_only.__class__.__name__

        image_size = int(titok.config.dataset.preprocessing.crop_size)
        example_input = load_image(image_paths[0], image_size).to("cpu")

        print("[3/8] Exporting encoder boundary")
        exported_program = export_encoder_program(encoder_only, example_input)

        print("[4/8] Preparing Arm Ethos-U PTQ observers")
        prepared_encoder, compile_spec = prepare_exported_encoder_for_ptq(
            exported_program,
            backend="ethosu",
            is_per_channel=args.per_channel,
            quantization_profile=args.quantization_profile,
            ethos_target=args.ethos_target,
            ethos_system_config=args.ethos_system_config,
            ethos_memory_mode=args.ethos_memory_mode,
            ethos_config_ini=args.ethos_config_ini,
            ethos_extra_flags=args.ethos_extra_flag,
            quantize_matmul=args.quantize_matmul,
        )
        summary["compile_spec_flags"] = compile_spec.compiler_flags if compile_spec is not None else None

        print(f"[5/8] Calibrating on {len(image_paths)} image(s)")
        calibrate_prepared_encoder(prepared_encoder, image_paths, image_size)

        print("[6/8] Converting quantized encoder")
        quantized_encoder = convert_encoder_after_ptq(prepared_encoder, backend="ethosu")

        final_export = torch.export.export(quantized_encoder, (example_input,), strict=True)
        capture_input = None
        capture_records = {}
        if capture_image_path is not None:
            print(f"[6b/8] Capturing pre-lowering outputs for {capture_image_path}")
            capture_input = load_image(capture_image_path, image_size).to("cpu")
            with torch.no_grad():
                quantized_module_output = quantized_encoder(capture_input).detach().to(
                    "cpu",
                    dtype=torch.float32,
                )
                final_export_output = final_export.module()(capture_input).detach().to(
                    "cpu",
                    dtype=torch.float32,
                )

            quantized_module_path = output_dir / f"{args.capture_output_prefix}_quantized_module_pre_lowering.npz"
            final_export_path = output_dir / f"{args.capture_output_prefix}_final_export_pre_rewrite.npz"
            save_npz_tensor(quantized_module_path, quantized_module_output)
            save_npz_tensor(final_export_path, final_export_output)
            capture_records["quantized_module_pre_lowering"] = {
                "path": str(quantized_module_path),
                "stats": tensor_stats(quantized_module_output),
            }
            capture_records["final_export_pre_rewrite"] = {
                "path": str(final_export_path),
                "stats": tensor_stats(final_export_output),
            }

        if args.rewrite_cortexm_bmm:
            print("[7/8] Rewriting BMM fallback islands to Cortex-M custom ops")
            summary["pre_rewrite_graph_summary"] = summarize_fx_graph(final_export, "pre_rewrite")
            lowered_export = rewrite_qdq_bmm_to_cortex_m(final_export)
            summary["post_rewrite_graph_summary"] = summarize_fx_graph(lowered_export, "post_rewrite")
        else:
            print("[7/8] Skipping Cortex-M BMM fallback rewrite")
            lowered_export = final_export
            summary["final_export_graph_summary"] = summarize_fx_graph(final_export, "final_export")

        if capture_image_path is not None:
            with torch.no_grad():
                lowering_input_output = lowered_export.module()(capture_input).detach().to(
                    "cpu",
                    dtype=torch.float32,
                )
            lowering_input_path = output_dir / f"{args.capture_output_prefix}_lowering_input_pre_partition.npz"
            capture_metadata_path = output_dir / f"{args.capture_output_prefix}_pre_lowering_capture.json"
            save_npz_tensor(lowering_input_path, lowering_input_output)
            capture_records["lowering_input_pre_partition"] = {
                "path": str(lowering_input_path),
                "stats": tensor_stats(lowering_input_output),
            }
            capture_report = {
                "image": str(capture_image_path),
                "capture_output_prefix": args.capture_output_prefix,
                "quantization_profile": args.quantization_profile,
                "encoder_variant": args.encoder_variant,
                "prefix_num_blocks": args.prefix_num_blocks,
                "rewrite_cortexm_bmm": args.rewrite_cortexm_bmm,
                "outputs": capture_records,
            }
            capture_metadata_path.write_text(json.dumps(capture_report, indent=2) + "\n")
            summary["pre_lowering_capture"] = {
                **capture_report,
                "metadata_path": str(capture_metadata_path),
            }

        print("[8/8] Attempting Arm Ethos-U lowering")
        partitioner = EthosUPartitioner(
            EthosUCompatCompileSpec(
                args.ethos_target,
                system_config=args.ethos_system_config,
                memory_mode=args.ethos_memory_mode,
                config_ini=args.ethos_config_ini,
                extra_flags=args.ethos_extra_flag,
            )
        )
        edge_manager = to_edge_transform_and_lower(lowered_export, partitioner=[partitioner])
        summary["post_partition_graph_summary"] = summarize_fx_graph(
            edge_manager.exported_program().graph_module,
            "post_partition",
        )
        if args.post_partition_qdq_out_fix:
            edge_manager = edge_manager.transform([ReplaceSurvivingQdqWithOutVarPass()])
            summary["post_partition_qdq_fix_graph_summary"] = summarize_fx_graph(
                edge_manager.exported_program().graph_module,
                "post_partition_qdq_fix",
            )
        methods = edge_manager.methods
        if hasattr(methods, "keys"):
            edge_methods = list(methods.keys())
        else:
            edge_methods = list(methods)
        summary["status"] = "lowering_succeeded"
        summary["edge_methods"] = edge_methods
        executorch_program = edge_manager.to_executorch()
        summary["runtime_program_summary"] = summarize_executorch_program(executorch_program)
        artifact_path.write_bytes(executorch_program.buffer)
        summary["pte_path"] = str(artifact_path)
        print(f"Lowering succeeded; summary written to {summary_path}")
    except Exception as exc:
        summary["status"] = "lowering_failed"
        summary["error_type"] = type(exc).__name__
        summary["error_message"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        print(f"Lowering failed with {type(exc).__name__}: {exc}")
        if args.strict_failure:
            summary_path.write_text(json.dumps(summary, indent=2))
            raise
    finally:
        summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
