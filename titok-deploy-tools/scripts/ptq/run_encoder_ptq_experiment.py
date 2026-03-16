import argparse
import json
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.ptq import (
    build_encoder_quantizer_split,
    calibrate_prepared_encoder,
    convert_encoder_after_ptq,
    export_encoder_program,
    load_manifest_records,
    prepare_exported_encoder_for_ptq,
    run_encoder_with_float_quantizer,
    save_token_records,
    summarize_token_records,
)
from titok_deploy_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.utils import load_image, resolve_input_path, resolve_output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run encoder-only PTQ prep for TiTok-S-128 while keeping the VQ quantizer float."
    )
    parser.add_argument("--titok-root", required=True, help="Path to a separate 1d-tokenizer checkout.")
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_s128_imagenet",
        help="Hugging Face repo for the pretrained TiTok-S-128 tokenizer.",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/ptq/calibration_manifest.json",
        help="Calibration manifest listing representative images.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ptq",
        help="Directory where PTQ artifacts will be written.",
    )
    parser.add_argument(
        "--encoder-artifact-name",
        default="titok_s128_encoder_only.pt2",
        help="Filename for the exported encoder-only torch.export artifact.",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Use per-channel symmetric quantization for supported weights.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Stop after prepare+calibration without converting the encoder graph.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    manifest_path = resolve_input_path(args.manifest, REPO_ROOT)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Calibration manifest not found: {manifest_path}")
    image_paths = load_manifest_records(manifest_path)
    if not image_paths:
        raise SystemExit("Calibration manifest is empty.")

    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
    encoder_artifact_path = output_dir / Path(args.encoder_artifact_name).name
    encoder_metadata_path = output_dir / "titok_s128_encoder_only_metadata.json"
    prepared_summary_path = output_dir / "s128_encoder_ptq_prepare_summary.json"
    converted_tokens_path = output_dir / "s128_encoder_ptq_tokens.json"
    converted_summary_path = output_dir / "s128_encoder_ptq_summary.json"

    print(f"[1/6] Loading TiTok model from {args.repo_id} on CPU")
    titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
    titok.requires_grad_(False)

    print("[2/6] Building encoder-only and float-VQ wrappers")
    encoder_only, latents_to_tokens, _ = build_encoder_quantizer_split(titok)
    encoder_only = encoder_only.eval().to("cpu")
    latents_to_tokens = latents_to_tokens.eval().to("cpu")
    encoder_only.requires_grad_(False)
    latents_to_tokens.requires_grad_(False)

    image_size = int(titok.config.dataset.preprocessing.crop_size)
    example_input = load_image(image_paths[0], image_size).to("cpu")

    print("[3/6] Exporting encoder boundary and preparing PTQ observers")
    exported_program = export_encoder_program(encoder_only, example_input)
    exported_encoder = exported_program.module()
    exported_latent = exported_encoder(example_input)
    torch.export.save(exported_program, encoder_artifact_path)
    encoder_metadata = {
        "repo_id": args.repo_id,
        "artifact_path": str(encoder_artifact_path),
        "image_size": image_size,
        "input_shape": list(example_input.shape),
        "output_shape": list(exported_latent.shape),
        "output_dtype": str(exported_latent.dtype),
        "quantizer_boundary": "encoder_only_quantized_vq_float",
    }
    encoder_metadata_path.write_text(json.dumps(encoder_metadata, indent=2))
    prepared_encoder = prepare_exported_encoder_for_ptq(
        exported_program,
        is_per_channel=args.per_channel,
    )

    print(f"[4/6] Calibrating encoder observers on {len(image_paths)} image(s)")
    calibrate_prepared_encoder(prepared_encoder, image_paths, image_size)

    prepare_summary = {
        "repo_id": args.repo_id,
        "manifest_path": str(manifest_path),
        "encoder_artifact_path": str(encoder_artifact_path),
        "num_calibration_images": len(image_paths),
        "image_size": image_size,
        "per_channel": args.per_channel,
        "skip_convert": args.skip_convert,
        "prepared_encoder_type": type(prepared_encoder).__name__,
        "quantizer_boundary": "encoder_only_quantized_vq_float",
    }
    prepared_summary_path.write_text(json.dumps(prepare_summary, indent=2))

    if args.skip_convert:
        print(f"[5/6] Skipping convert; saved prepare summary to {prepared_summary_path}")
        print("[6/6] PTQ prepare stage complete")
        return

    print("[5/6] Converting calibrated encoder to a quantized graph")
    quantized_encoder = convert_encoder_after_ptq(prepared_encoder)

    print("[6/6] Running quantized encoder + float VQ on the calibration set")
    records = []
    token_shape = None
    with torch.no_grad():
        for image_path in image_paths:
            image = load_image(image_path, image_size).to("cpu")
            tokens = run_encoder_with_float_quantizer(quantized_encoder, latents_to_tokens, image)
            tokens = tokens.to("cpu", dtype=torch.int64)
            token_shape = list(tokens.shape)
            records.append(
                {
                    "image": str(image_path),
                    "tokens": tokens[0].tolist(),
                }
            )

    save_token_records(
        converted_tokens_path,
        records,
        repo_id=args.repo_id,
        image_size=image_size,
        token_shape=token_shape,
        metadata={
            "manifest_path": str(manifest_path),
            "source": "encoder_ptq_float_vq",
            "per_channel": args.per_channel,
        },
    )
    converted_summary = summarize_token_records(records) | {
        "repo_id": args.repo_id,
        "manifest_path": str(manifest_path),
        "prepare_summary_path": str(prepared_summary_path),
        "quantizer_boundary": "encoder_only_quantized_vq_float",
        "per_channel": args.per_channel,
    }
    converted_summary_path.write_text(json.dumps(converted_summary, indent=2))
    print(f"Saved PTQ token outputs to {converted_tokens_path}")


if __name__ == "__main__":
    main()
