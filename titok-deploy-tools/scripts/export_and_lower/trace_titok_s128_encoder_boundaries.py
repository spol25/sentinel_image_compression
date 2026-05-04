import argparse
import json
from collections import Counter
from pathlib import Path
import sys
import traceback

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
EXECUTORCH_SRC_ROOT = WORKSPACE_ROOT / "executorch-main" / "src"
SRC_ROOT = REPO_ROOT / "src"

for path in (EXECUTORCH_SRC_ROOT, SRC_ROOT):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

import torch
from executorch.backends.arm.ethosu import EthosUPartitioner
from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import ToOutVarPass

from titok_deploy_tools.lowering_tools.ethosu_compat import EthosUCompatCompileSpec
from titok_deploy_tools.lowering_tools.executorch_summary import summarize_executorch_program
from titok_deploy_tools.lowering_tools.graph_summary import summarize_fx_graph
from titok_deploy_tools.ptq_tools.ptq import (
    build_encoder_only_wrapper,
    calibrate_prepared_encoder,
    convert_encoder_after_ptq,
    export_encoder_program,
    load_manifest_records,
    prepare_exported_encoder_for_ptq,
)
from titok_deploy_tools.lowering_tools.post_partition_qdq_fix import ReplaceSurvivingQdqWithOutVarPass
from titok_deploy_tools.wrapper_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.wrapper_tools.utils import load_image, resolve_input_path, resolve_output_dir


ENCODER_VARIANTS = (
    "baseline",
    "einsum_attention",
    "reshape_batch",
    "bmm_attention",
    "source_matmul_attention",
    "source_sdpa_attention",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace two TiTok-S-128 encoder variants and save per-boundary op diffs."
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
        help="Calibration manifest used for PTQ preparation.",
    )
    parser.add_argument(
        "--calibration-count",
        type=int,
        default=4,
        help="How many calibration images to use.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/encoder_boundary_compare",
        help="Directory where variant boundary summaries will be written.",
    )
    parser.add_argument(
        "--summary-name",
        default="encoder_boundary_compare_summary.json",
        help="Filename for the aggregate comparison JSON.",
    )
    parser.add_argument(
        "--baseline-variant",
        choices=ENCODER_VARIANTS,
        default="baseline",
        help="Reference encoder wrapper to trace.",
    )
    parser.add_argument(
        "--compare-variant",
        choices=ENCODER_VARIANTS,
        default="einsum_attention",
        help="Comparison encoder wrapper to trace.",
    )
    parser.add_argument(
        "--ethos-target",
        default="ethos-u65-256",
        help="Ethos-U accelerator target string.",
    )
    parser.add_argument(
        "--ethos-system-config",
        default="Ethos_U65_High_End",
        help="Optional Vela system_config override.",
    )
    parser.add_argument(
        "--ethos-memory-mode",
        default="Shared_Sram",
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
        help="Additional Vela flag to pass through the Ethos-U compile spec. Repeat for multiple flags.",
    )
    parser.add_argument(
        "--dump-vela-intermediates",
        action="store_true",
        help="Preserve Vela TOSA/NPZ/debug artifacts under each variant output directory.",
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
        help="Exit non-zero if either variant fails after writing summaries.",
    )
    parser.add_argument(
        "--skip-post-partition-qdq-out-fix",
        action="store_true",
        help="Do not rewrite surviving post-partition q/dq ops to .out variants before to_executorch().",
    )
    parser.add_argument(
        "--quantize-matmul",
        action="store_true",
        help="Ask the Ethos-U quantizer to quantize aten.matmul nodes when supported.",
    )
    return parser.parse_args()


def _variant_output_name(variant_name: str) -> str:
    return variant_name.replace("/", "_").replace(" ", "_")


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=2))
    return str(path)


def _counter_from_mapping(mapping: dict[str, int]) -> Counter[str]:
    return Counter({str(k): int(v) for k, v in mapping.items()})


def _counter_from_runtime_list(items: list[dict]) -> Counter[str]:
    return Counter({item["name"]: int(item["count"]) for item in items})


def _top_count_deltas(baseline: Counter[str], variant: Counter[str], limit: int = 40) -> list[dict]:
    names = sorted(set(baseline) | set(variant))
    rows = []
    for name in names:
        base = baseline.get(name, 0)
        var = variant.get(name, 0)
        delta = var - base
        if delta:
            rows.append(
                {
                    "name": name,
                    "baseline_count": base,
                    "variant_count": var,
                    "delta": delta,
                }
            )
    rows.sort(key=lambda row: (-abs(row["delta"]), row["name"]))
    return rows[:limit]


def _compare_stage_summaries(baseline_summary: dict, variant_summary: dict) -> dict:
    result = {
        "baseline_stage": baseline_summary["stage"],
        "variant_stage": variant_summary["stage"],
        "boundary_type": baseline_summary.get("boundary_type", variant_summary.get("boundary_type")),
    }

    if result["boundary_type"] == "fx_graph":
        baseline_counts = _counter_from_mapping(baseline_summary["target_counts"])
        variant_counts = _counter_from_mapping(variant_summary["target_counts"])
        result.update(
            {
                "same_graph_signature": baseline_summary.get("graph_signature")
                == variant_summary.get("graph_signature"),
                "baseline_node_count": baseline_summary["node_count"],
                "variant_node_count": variant_summary["node_count"],
                "baseline_unique_op_count": baseline_summary["unique_op_count"],
                "variant_unique_op_count": variant_summary["unique_op_count"],
                "count_deltas": _top_count_deltas(baseline_counts, variant_counts),
            }
        )
        return result

    baseline_kernel_counts = _counter_from_runtime_list(baseline_summary["kernel_op_counts"])
    variant_kernel_counts = _counter_from_runtime_list(variant_summary["kernel_op_counts"])
    baseline_delegate_counts = _counter_from_runtime_list(baseline_summary["delegate_call_counts"])
    variant_delegate_counts = _counter_from_runtime_list(variant_summary["delegate_call_counts"])
    result.update(
        {
            "baseline_instruction_counts": baseline_summary["instruction_counts"],
            "variant_instruction_counts": variant_summary["instruction_counts"],
            "kernel_count_deltas": _top_count_deltas(baseline_kernel_counts, variant_kernel_counts),
            "delegate_count_deltas": _top_count_deltas(baseline_delegate_counts, variant_delegate_counts),
        }
    )
    return result


def trace_variant(
    *,
    variant_name: str,
    titok,
    example_input: torch.Tensor,
    image_paths: list[Path],
    image_size: int,
    args,
    variant_dir: Path,
) -> dict:
    variant_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant_name": variant_name,
        "status": "started",
        "stages": {},
    }

    try:
        wrapper = build_encoder_only_wrapper(titok, encoder_variant=variant_name).eval().to("cpu")
        wrapper.requires_grad_(False)
        payload["wrapper_variant"] = wrapper.__class__.__name__

        exported_program = export_encoder_program(wrapper, example_input)
        export_summary = summarize_fx_graph(exported_program, "01_export")
        payload["stages"]["01_export"] = {
            "summary_path": _write_json(variant_dir / "01_export.json", export_summary),
            "node_count": export_summary["node_count"],
            "unique_op_count": export_summary["unique_op_count"],
        }

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
        payload["compile_spec_flags"] = compile_spec.compiler_flags if compile_spec is not None else None

        prepare_summary = summarize_fx_graph(prepared_encoder, "02_prepare")
        payload["stages"]["02_prepare"] = {
            "summary_path": _write_json(variant_dir / "02_prepare.json", prepare_summary),
            "node_count": prepare_summary["node_count"],
            "unique_op_count": prepare_summary["unique_op_count"],
        }

        calibrate_prepared_encoder(prepared_encoder, image_paths, image_size)
        post_calibration_summary = summarize_fx_graph(prepared_encoder, "03_post_calibration")
        payload["stages"]["03_post_calibration"] = {
            "summary_path": _write_json(variant_dir / "03_post_calibration.json", post_calibration_summary),
            "node_count": post_calibration_summary["node_count"],
            "unique_op_count": post_calibration_summary["unique_op_count"],
        }

        quantized_encoder = convert_encoder_after_ptq(prepared_encoder, backend="ethosu")
        convert_summary = summarize_fx_graph(quantized_encoder, "04_convert")
        payload["stages"]["04_convert"] = {
            "summary_path": _write_json(variant_dir / "04_convert.json", convert_summary),
            "node_count": convert_summary["node_count"],
            "unique_op_count": convert_summary["unique_op_count"],
        }

        final_export = torch.export.export(quantized_encoder, (example_input,), strict=True)
        final_export_summary = summarize_fx_graph(final_export, "05_final_export")
        payload["stages"]["05_final_export"] = {
            "summary_path": _write_json(variant_dir / "05_final_export.json", final_export_summary),
            "node_count": final_export_summary["node_count"],
            "unique_op_count": final_export_summary["unique_op_count"],
        }

        lowering_compile_spec = EthosUCompatCompileSpec(
            args.ethos_target,
            system_config=args.ethos_system_config,
            memory_mode=args.ethos_memory_mode,
            config_ini=args.ethos_config_ini,
            extra_flags=args.ethos_extra_flag,
        )
        if args.dump_vela_intermediates:
            vela_dir = variant_dir / "vela_intermediates"
            vela_dir.mkdir(parents=True, exist_ok=True)
            lowering_compile_spec.dump_intermediate_artifacts_to(str(vela_dir))
            payload["vela_intermediate_dir"] = str(vela_dir)
        payload["lowering_compile_spec_flags"] = lowering_compile_spec.compiler_flags

        partitioner = EthosUPartitioner(lowering_compile_spec)
        edge_manager = to_edge_transform_and_lower(final_export, partitioner=[partitioner])

        edge_summary = summarize_fx_graph(edge_manager.exported_program(), "06_edge_lowered")
        payload["stages"]["06_edge_lowered"] = {
            "summary_path": _write_json(variant_dir / "06_edge_lowered.json", edge_summary),
            "node_count": edge_summary["node_count"],
            "unique_op_count": edge_summary["unique_op_count"],
        }

        to_executorch_config = None
        if not args.skip_post_partition_qdq_out_fix:
            try:
                edge_manager = edge_manager.transform([ReplaceSurvivingQdqWithOutVarPass()])
            except Exception as exc:
                payload["post_partition_qdq_fix_fallback"] = {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "strategy": "to_executorch_ignore_missing_qdq_out_variants",
                }
                to_executorch_config = ExecutorchBackendConfig(
                    to_out_var_pass=ToOutVarPass(ignore_to_out_var_failure=True)
                )
            qdq_fix_summary = summarize_fx_graph(edge_manager.exported_program(), "06b_qdq_out_fixed")
            payload["stages"]["06b_qdq_out_fixed"] = {
                "summary_path": _write_json(variant_dir / "06b_qdq_out_fixed.json", qdq_fix_summary),
                "node_count": qdq_fix_summary["node_count"],
                "unique_op_count": qdq_fix_summary["unique_op_count"],
            }

        executorch_program = edge_manager.to_executorch(config=to_executorch_config)
        executorch_summary = summarize_executorch_program(executorch_program)
        executorch_summary["stage"] = "07_executorch_runtime"
        executorch_summary["boundary_type"] = "executorch_program"
        payload["stages"]["07_executorch_runtime"] = {
            "summary_path": _write_json(variant_dir / "07_executorch_runtime.json", executorch_summary),
            "instruction_counts": executorch_summary["instruction_counts"],
            "kernel_unique_op_count": len(executorch_summary["kernel_op_counts"]),
            "delegate_unique_op_count": len(executorch_summary["delegate_call_counts"]),
        }

        artifact_path = variant_dir / f"{_variant_output_name(variant_name)}.pte"
        artifact_path.write_bytes(executorch_program.buffer)
        payload["pte_path"] = str(artifact_path)
        payload["status"] = "lowering_succeeded"
    except Exception as exc:
        payload["status"] = "lowering_failed"
        payload["error_type"] = type(exc).__name__
        payload["error_message"] = str(exc)
        payload["traceback"] = traceback.format_exc()

    _write_json(variant_dir / "variant_summary.json", payload)
    return payload


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
    diff_dir = output_dir / "diffs"
    diff_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "repo_id": args.repo_id,
        "manifest_path": str(manifest_path),
        "num_calibration_images": len(image_paths),
        "ethos_target": args.ethos_target,
        "ethos_system_config": args.ethos_system_config,
        "ethos_memory_mode": args.ethos_memory_mode,
        "ethos_config_ini": args.ethos_config_ini,
        "ethos_extra_flags": args.ethos_extra_flag,
        "dump_vela_intermediates": args.dump_vela_intermediates,
        "baseline_variant": args.baseline_variant,
        "compare_variant": args.compare_variant,
        "quantize_matmul": args.quantize_matmul,
        "per_channel": args.per_channel,
        "quantization_profile": args.quantization_profile,
        "variants": {},
        "stage_diffs": {},
        "status": "started",
    }

    titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
    titok.requires_grad_(False)
    image_size = int(titok.config.dataset.preprocessing.crop_size)
    example_input = load_image(image_paths[0], image_size).to("cpu")

    print(f"[1/3] Tracing {args.baseline_variant} boundary flow")
    baseline = trace_variant(
        variant_name=args.baseline_variant,
        titok=titok,
        example_input=example_input,
        image_paths=image_paths,
        image_size=image_size,
        args=args,
        variant_dir=output_dir / _variant_output_name(args.baseline_variant),
    )
    payload["variants"][args.baseline_variant] = baseline

    print(f"[2/3] Tracing {args.compare_variant} boundary flow")
    compare = trace_variant(
        variant_name=args.compare_variant,
        titok=titok,
        example_input=example_input,
        image_paths=image_paths,
        image_size=image_size,
        args=args,
        variant_dir=output_dir / _variant_output_name(args.compare_variant),
    )
    payload["variants"][args.compare_variant] = compare

    print("[3/3] Writing stage-by-stage diffs")
    baseline_stage_names = set(baseline["stages"])
    compare_stage_names = set(compare["stages"])
    common_stage_names = sorted(baseline_stage_names & compare_stage_names)
    payload["common_stage_names"] = common_stage_names
    payload["baseline_only_stage_names"] = sorted(baseline_stage_names - compare_stage_names)
    payload["compare_only_stage_names"] = sorted(compare_stage_names - baseline_stage_names)

    for stage_name in common_stage_names:
        baseline_summary_path = Path(baseline["stages"][stage_name]["summary_path"])
        compare_summary_path = Path(compare["stages"][stage_name]["summary_path"])
        baseline_summary = json.loads(baseline_summary_path.read_text())
        compare_summary = json.loads(compare_summary_path.read_text())
        diff_payload = _compare_stage_summaries(baseline_summary, compare_summary)
        diff_payload["stage_name"] = stage_name
        diff_payload["baseline_variant"] = args.baseline_variant
        diff_payload["compare_variant"] = args.compare_variant
        diff_payload["baseline_summary_path"] = str(baseline_summary_path)
        diff_payload["compare_summary_path"] = str(compare_summary_path)
        diff_path = diff_dir / f"{stage_name}.json"
        _write_json(diff_path, diff_payload)
        payload["stage_diffs"][stage_name] = str(diff_path)

    if baseline["status"] == "lowering_succeeded" and compare["status"] == "lowering_succeeded":
        payload["status"] = "both_lowering_succeeded"
    elif baseline["status"] == "lowering_succeeded" and compare["status"] == "lowering_failed":
        payload["status"] = "compare_failed"
    elif baseline["status"] == "lowering_failed" and compare["status"] == "lowering_succeeded":
        payload["status"] = "baseline_failed"
    else:
        payload["status"] = "both_failed"

    _write_json(summary_path, payload)
    print(f"Wrote encoder boundary comparison to {summary_path}")

    if args.strict_failure and payload["status"] != "both_lowering_succeeded":
        raise SystemExit(payload["status"])


if __name__ == "__main__":
    main()
