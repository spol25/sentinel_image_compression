import argparse
import json
from pathlib import Path
import sys
import traceback

import torch
from executorch.backends.arm.ethosu import EthosUPartitioner
from executorch.exir import to_edge_transform_and_lower

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.ethosu_compat import EthosUCompatCompileSpec
from titok_deploy_tools.ptq import (
    calibrate_prepared_encoder,
    convert_encoder_after_ptq,
    export_encoder_program,
    load_manifest_records,
    prepare_exported_encoder_for_ptq,
)
from titok_deploy_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.utils import load_image, resolve_input_path, resolve_output_dir
from titok_deploy_tools.wrappers import TiTokEncoderPrefix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bisect Arm Ethos-U U65 lowering block-by-block for the TiTok-S-128 encoder."
    )
    parser.add_argument("--titok-root", required=True, help="Path to a separate 1d-tokenizer checkout.")
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_s128_imagenet",
        help="Hugging Face repo for the pretrained TiTok-S-128 tokenizer.",
    )
    parser.add_argument("--manifest", required=True, help="Calibration manifest used for PTQ preparation.")
    parser.add_argument(
        "--calibration-count",
        type=int,
        default=4,
        help="How many manifest images to use for the quick lowering bisect.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/arm_block_bisect",
        help="Directory where the bisect summary will be written.",
    )
    parser.add_argument(
        "--summary-name",
        default="ethosu_u65_block_bisect_summary.json",
        help="Filename for the bisect summary JSON.",
    )
    parser.add_argument(
        "--ethos-target",
        default="ethos-u65-256",
        help="Ethos-U accelerator target string.",
    )
    parser.add_argument("--ethos-system-config", default=None, help="Optional Vela system_config override.")
    parser.add_argument("--ethos-memory-mode", default=None, help="Optional Vela memory_mode override.")
    parser.add_argument(
        "--ethos-config-ini",
        default="Arm/vela.ini",
        help="Path to the Vela .ini file used in the compile spec.",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Use per-channel symmetric quantization for supported weights.",
    )
    parser.add_argument(
        "--stop-on-first-failure",
        action="store_true",
        help="Stop once the first failing block depth is identified.",
    )
    return parser.parse_args()


def attempt_lowering(
    titok,
    image_paths,
    image_size,
    num_blocks,
    *,
    per_channel,
    ethos_target,
    ethos_system_config,
    ethos_memory_mode,
    ethos_config_ini,
):
    encoder_prefix = TiTokEncoderPrefix(titok, num_blocks=num_blocks).eval().to("cpu")
    encoder_prefix.requires_grad_(False)
    example_input = load_image(image_paths[0], image_size).to("cpu")

    exported_program = export_encoder_program(encoder_prefix, example_input)
    prepared_encoder, compile_spec = prepare_exported_encoder_for_ptq(
        exported_program,
        backend="ethosu",
        is_per_channel=per_channel,
        ethos_target=ethos_target,
        ethos_system_config=ethos_system_config,
        ethos_memory_mode=ethos_memory_mode,
        ethos_config_ini=ethos_config_ini,
    )
    calibrate_prepared_encoder(prepared_encoder, image_paths, image_size)
    quantized_encoder = convert_encoder_after_ptq(prepared_encoder, backend="ethosu")
    final_export = torch.export.export(quantized_encoder, (example_input,), strict=True)
    partitioner = EthosUPartitioner(
        EthosUCompatCompileSpec(
            ethos_target,
            system_config=ethos_system_config,
            memory_mode=ethos_memory_mode,
            config_ini=ethos_config_ini,
        )
    )
    edge_manager = to_edge_transform_and_lower(final_export, partitioner=[partitioner])
    return {
        "status": "lowering_succeeded",
        "compile_spec_flags": compile_spec.compiler_flags if compile_spec is not None else None,
        "edge_methods": list(edge_manager.methods),
    }


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

    print("Loading TiTok model on CPU")
    titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
    titok.requires_grad_(False)
    image_size = int(titok.config.dataset.preprocessing.crop_size)
    total_blocks = titok.encoder.num_layers

    summary = {
        "repo_id": args.repo_id,
        "manifest_path": str(manifest_path),
        "num_calibration_images": len(image_paths),
        "ethos_target": args.ethos_target,
        "per_channel": args.per_channel,
        "total_blocks": total_blocks,
        "results": [],
    }

    first_failure = None
    for num_blocks in range(total_blocks + 1):
        print(f"[{num_blocks}/{total_blocks}] Attempting lowering with {num_blocks} block(s)")
        result = {"num_blocks": num_blocks}
        try:
            result.update(
                attempt_lowering(
                    titok,
                    image_paths,
                    image_size,
                    num_blocks,
                    per_channel=args.per_channel,
                    ethos_target=args.ethos_target,
                    ethos_system_config=args.ethos_system_config,
                    ethos_memory_mode=args.ethos_memory_mode,
                    ethos_config_ini=args.ethos_config_ini,
                )
            )
        except Exception as exc:
            result["status"] = "lowering_failed"
            result["error_type"] = type(exc).__name__
            result["error_message"] = str(exc)
            result["traceback"] = traceback.format_exc()
            if first_failure is None:
                first_failure = num_blocks
        summary["results"].append(result)
        summary["first_failure_block"] = first_failure
        summary["successful_blocks"] = [
            entry["num_blocks"] for entry in summary["results"] if entry["status"] == "lowering_succeeded"
        ]
        summary["failed_blocks"] = [
            entry["num_blocks"] for entry in summary["results"] if entry["status"] == "lowering_failed"
        ]
        summary_path.write_text(json.dumps(summary, indent=2))
        if args.stop_on_first_failure and first_failure is not None:
            break

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved block bisect summary to {summary_path}")


if __name__ == "__main__":
    main()
