import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

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
    export_quantized_encoder_program,
    load_manifest_records,
    prepare_exported_encoder_for_ptq,
)
from titok_deploy_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.utils import load_image, resolve_input_path, resolve_output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a shareable TiTok-S-128 POC bundle with quantized encoder, float VQ, decoder weights, and a minimal repro."
    )
    parser.add_argument("--titok-root", required=True, help="Path to the 1d-tokenizer checkout.")
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_s128_imagenet",
        help="Hugging Face repo id for the pretrained TiTok-S-128 tokenizer.",
    )
    parser.add_argument(
        "--manifest",
        default="/Volumes/Media/snapshot_serengeti_ptq_quickstart/manifests/calibration_manifest.json",
        help="Calibration manifest used to prepare the quantized encoder.",
    )
    parser.add_argument(
        "--example-image",
        default=None,
        help="Example image to include in the bundle. Defaults to the first calibration image, or a TiTok asset as fallback.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/poc_bundle/titok_s128_ethosu_u65",
        help="Directory where the bundle will be written.",
    )
    parser.add_argument(
        "--max-calibration-images",
        type=int,
        default=None,
        help="Optional cap on the number of calibration images to use.",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Use per-channel symmetric quantization where supported.",
    )
    parser.add_argument(
        "--quantizer-backend",
        choices=("xnnpack", "ethosu"),
        default="ethosu",
        help="PTQ backend to use for the quantized encoder handoff.",
    )
    parser.add_argument(
        "--ethos-target",
        default="ethos-u65-256",
        help="Ethos-U accelerator target when using the Arm Ethos-U quantizer.",
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
        help="Path to the Vela .ini file used in the Ethos-U compile spec.",
    )
    return parser.parse_args()


def choose_example_image(args_example_image: str | None, titok_root: Path, calibration_images: list[Path]) -> Path:
    if args_example_image is not None:
        return Path(args_example_image)
    if calibration_images:
        return calibration_images[0]
    return titok_root / "assets" / "ILSVRC2012_val_00010240.png"


def fallback_calibration_images(titok_root: Path) -> list[Path]:
    fallback = [
        titok_root / "assets" / "ILSVRC2012_val_00010240.png",
        titok_root / "assets" / "ILSVRC2012_val_00008636.png",
    ]
    return [path for path in fallback if path.exists()]


def write_minimal_repro(
    output_path: Path,
    *,
    repo_root: Path,
    titok_root: Path,
    bundle_dir: Path,
    image_size: int,
    quantizer_backend: str,
    per_channel: bool,
    ethos_target: str,
    ethos_system_config: str | None,
    ethos_memory_mode: str | None,
    ethos_config_ini: str | None,
):
    rel_bundle = bundle_dir.relative_to(repo_root)
    lines = [
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        "import torch",
        "from omegaconf import OmegaConf",
        "",
        f"REPO_ROOT = Path({str(repo_root)!r})",
        f"TITOK_ROOT = Path({str(titok_root)!r})",
        f"BUNDLE_DIR = REPO_ROOT / {str(rel_bundle)!r}",
        "",
        "SRC_ROOT = REPO_ROOT / 'src'",
        "if str(SRC_ROOT) not in sys.path:",
        "    sys.path.insert(0, str(SRC_ROOT))",
        "if str(TITOK_ROOT) not in sys.path:",
        "    sys.path.insert(0, str(TITOK_ROOT))",
        "",
        "from modeling.titok import TiTok",
        "import torch.ao.quantization.fx._decomposed  # registers quantized_decomposed ops for pt2 load",
        "from titok_deploy_tools.ptq import build_encoder_quantizer_split",
        "from titok_deploy_tools.utils import load_image, save_reconstruction",
        "",
        "config = OmegaConf.create(json.loads((BUNDLE_DIR / 'titok_s128_config.json').read_text()))",
        "titok = TiTok(config).eval().to('cpu')",
        "titok.requires_grad_(False)",
        "titok.encoder.load_state_dict(torch.load(BUNDLE_DIR / 'float_encoder_state_dict.pt', map_location='cpu'))",
        "latent_tokens = torch.load(BUNDLE_DIR / 'latent_tokens.pt', map_location='cpu')",
        "titok.latent_tokens.data.copy_(latent_tokens)",
        "titok.quantize.load_state_dict(torch.load(BUNDLE_DIR / 'vq_quantizer_state_dict.pt', map_location='cpu'))",
        "titok.decoder.load_state_dict(torch.load(BUNDLE_DIR / 'decoder_state_dict.pt', map_location='cpu'))",
        "if hasattr(titok, 'pixel_quantize') and (BUNDLE_DIR / 'pixel_quantizer_state_dict.pt').exists():",
        "    titok.pixel_quantize.load_state_dict(torch.load(BUNDLE_DIR / 'pixel_quantizer_state_dict.pt', map_location='cpu'))",
        "if hasattr(titok, 'pixel_decoder') and (BUNDLE_DIR / 'pixel_decoder_state_dict.pt').exists():",
        "    titok.pixel_decoder.load_state_dict(torch.load(BUNDLE_DIR / 'pixel_decoder_state_dict.pt', map_location='cpu'))",
        "",
        "encoder_only, latents_to_tokens, _ = build_encoder_quantizer_split(titok)",
        "quantized_encoder = torch.export.load(BUNDLE_DIR / 'quantized_encoder.pt2').module()",
        "",
        "with torch.no_grad():",
        "    image = load_image(BUNDLE_DIR / 'example_image.png', image_size=" + str(image_size) + ")",
        "    float_latent = encoder_only(image)",
        "    float_tokens = latents_to_tokens(float_latent)",
        "    float_reconstruction = titok.decode_tokens(float_tokens.unsqueeze(1) if float_tokens.ndim == 2 else float_tokens)",
        "    latent = quantized_encoder(image)",
        "    tokens = latents_to_tokens(latent)",
        "    quantized_reconstruction = titok.decode_tokens(tokens.unsqueeze(1) if tokens.ndim == 2 else tokens)",
        "",
        "save_reconstruction(float_reconstruction, BUNDLE_DIR / 'reconstruction_float_encoder.png')",
        "save_reconstruction(quantized_reconstruction, BUNDLE_DIR / 'reconstruction_quantized_encoder.png')",
        "",
        "print('tokens_shape:', tuple(tokens.shape))",
        "print('tokens_sample:', tokens[0, :16].tolist() if tokens.ndim == 2 else tokens.reshape(tokens.shape[0], -1)[0, :16].tolist())",
        "print('float_reconstruction_shape:', tuple(float_reconstruction.shape))",
        "print('quantized_reconstruction_shape:', tuple(quantized_reconstruction.shape))",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    titok_root = add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    manifest_path = resolve_input_path(args.manifest, REPO_ROOT)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Calibration manifest not found: {manifest_path}")

    requested_calibration_images = load_manifest_records(manifest_path)
    calibration_images = [path for path in requested_calibration_images if path.exists()]
    missing_calibration_images = len(requested_calibration_images) - len(calibration_images)
    if args.max_calibration_images is not None:
        calibration_images = calibration_images[: args.max_calibration_images]
    if not calibration_images:
        calibration_images = fallback_calibration_images(titok_root)
    if args.max_calibration_images is not None:
        calibration_images = calibration_images[: args.max_calibration_images]
    if not calibration_images:
        raise SystemExit("No calibration images available.")

    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
    bundle_dir = output_dir

    print(f"[1/6] Loading TiTok model from {args.repo_id}")
    titok = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
    titok.requires_grad_(False)

    print("[2/6] Building encoder/VQ split and example input")
    encoder_only, latents_to_tokens, _ = build_encoder_quantizer_split(titok)
    encoder_only = encoder_only.eval().to("cpu")
    latents_to_tokens = latents_to_tokens.eval().to("cpu")
    image_size = int(titok.config.dataset.preprocessing.crop_size)
    example_image_path = choose_example_image(args.example_image, titok_root, calibration_images)
    example_input = load_image(example_image_path, image_size).to("cpu")
    if missing_calibration_images:
        print(
            f"Warning: {missing_calibration_images} calibration image(s) from the manifest were missing locally; using {len(calibration_images)} available image(s) instead."
        )

    print("[3/6] Exporting encoder boundary and running PTQ convert")
    exported_program = export_encoder_program(encoder_only, example_input)
    prepared_encoder, compile_spec = prepare_exported_encoder_for_ptq(
        exported_program,
        backend=args.quantizer_backend,
        is_per_channel=args.per_channel,
        ethos_target=args.ethos_target,
        ethos_system_config=args.ethos_system_config,
        ethos_memory_mode=args.ethos_memory_mode,
        ethos_config_ini=args.ethos_config_ini,
    )
    calibrate_prepared_encoder(prepared_encoder, calibration_images, image_size)
    quantized_encoder = convert_encoder_after_ptq(prepared_encoder, backend=args.quantizer_backend)

    print("[4/6] Saving quantized encoder and float decode path weights")
    torch.save(quantized_encoder.state_dict(), bundle_dir / "quantized_encoder_state_dict.pt")
    quantized_exported_program = export_quantized_encoder_program(quantized_encoder, example_input)
    torch.export.save(quantized_exported_program, bundle_dir / "quantized_encoder.pt2")
    torch.export.save(exported_program, bundle_dir / "encoder_boundary_export.pt2")
    torch.save(titok.encoder.state_dict(), bundle_dir / "float_encoder_state_dict.pt")
    torch.save(titok.latent_tokens.detach().cpu(), bundle_dir / "latent_tokens.pt")
    torch.save(latents_to_tokens.quantize.state_dict(), bundle_dir / "vq_quantizer_state_dict.pt")
    torch.save(titok.decoder.state_dict(), bundle_dir / "decoder_state_dict.pt")
    if hasattr(titok, "pixel_quantize"):
        torch.save(titok.pixel_quantize.state_dict(), bundle_dir / "pixel_quantizer_state_dict.pt")
    if hasattr(titok, "pixel_decoder"):
        torch.save(titok.pixel_decoder.state_dict(), bundle_dir / "pixel_decoder_state_dict.pt")

    print("[5/6] Copying example assets and writing metadata")
    shutil.copy2(example_image_path, bundle_dir / "example_image.png")
    shutil.copy2(manifest_path, bundle_dir / "calibration_manifest.json")
    config_dict = OmegaConf.to_container(titok.config, resolve=True)
    (bundle_dir / "titok_s128_config.json").write_text(json.dumps(config_dict, indent=2))

    with torch.no_grad():
        latent = quantized_encoder(example_input)
        tokens = latents_to_tokens(latent).to("cpu", dtype=torch.int64)

    metadata = {
        "repo_id": args.repo_id,
        "bundle_dir": str(bundle_dir),
        "titok_root": str(titok_root),
        "manifest_path": str(manifest_path),
        "requested_calibration_images": len(requested_calibration_images),
        "num_calibration_images": len(calibration_images),
        "missing_calibration_images": missing_calibration_images,
        "calibration_images_used": [str(path) for path in calibration_images],
        "example_image": str(example_image_path),
        "image_size": image_size,
        "quantizer_backend": args.quantizer_backend,
        "per_channel": args.per_channel,
        "quantized_encoder_state_dict": str(bundle_dir / "quantized_encoder_state_dict.pt"),
        "quantized_encoder_pt2": str(bundle_dir / "quantized_encoder.pt2"),
        "encoder_boundary_export": str(bundle_dir / "encoder_boundary_export.pt2"),
        "float_encoder_state_dict": str(bundle_dir / "float_encoder_state_dict.pt"),
        "latent_tokens": str(bundle_dir / "latent_tokens.pt"),
        "vq_quantizer_state_dict": str(bundle_dir / "vq_quantizer_state_dict.pt"),
        "decoder_state_dict": str(bundle_dir / "decoder_state_dict.pt"),
        "pixel_quantizer_state_dict": str(bundle_dir / "pixel_quantizer_state_dict.pt") if hasattr(titok, "pixel_quantize") else None,
        "pixel_decoder_state_dict": str(bundle_dir / "pixel_decoder_state_dict.pt") if hasattr(titok, "pixel_decoder") else None,
        "example_tokens_shape": list(tokens.shape),
        "example_tokens_first16": tokens.reshape(tokens.shape[0], -1)[0, :16].tolist(),
        "quantizer_boundary": "encoder_only_quantized_vq_float_decoder_float",
    }
    if compile_spec is not None:
        metadata["compile_spec_target"] = compile_spec.target
        metadata["compile_spec_flags"] = compile_spec.compiler_flags
    (bundle_dir / "bundle_metadata.json").write_text(json.dumps(metadata, indent=2))

    write_minimal_repro(
        bundle_dir / "minimal_titok_s128_handoff.py",
        repo_root=REPO_ROOT,
        titok_root=titok_root,
        bundle_dir=bundle_dir,
        image_size=image_size,
        quantizer_backend=args.quantizer_backend,
        per_channel=args.per_channel,
        ethos_target=args.ethos_target,
        ethos_system_config=args.ethos_system_config,
        ethos_memory_mode=args.ethos_memory_mode,
        ethos_config_ini=args.ethos_config_ini,
    )

    readme = "\n".join(
        [
            "# TiTok-S-128 Ethos-U65 POC Bundle",
            "",
            "Files in this directory:",
            "- `quantized_encoder_state_dict.pt`: PTQ-converted encoder-only graph weights.",
            "- `quantized_encoder.pt2`: PT2 artifact for the PTQ-converted encoder-only graph.",
            "- `encoder_boundary_export.pt2`: float encoder-only export at the encoder/VQ split.",
            "- `float_encoder_state_dict.pt`: pretrained float encoder weights.",
            "- `latent_tokens.pt`: pretrained learned latent-token parameters used by the encoder.",
            "- `vq_quantizer_state_dict.pt`: float TiTok VQ quantizer weights.",
            "- `decoder_state_dict.pt`: TiTok decoder weights.",
            "- `pixel_quantizer_state_dict.pt`: pixel-space quantizer used by the decode path when finetune_decoder is enabled.",
            "- `pixel_decoder_state_dict.pt`: pixel-space decoder used by the decode path when finetune_decoder is enabled.",
            "- `example_image.png`: example image included for quick testing.",
            "- `calibration_manifest.json`: manifest used for encoder PTQ calibration.",
            "- `titok_s128_config.json`: model config needed to reconstruct the TiTok modules.",
            "- `minimal_titok_s128_handoff.py`: stripped down script showing how to load the bundle.",
            "- `bundle_metadata.json`: summary of the exact settings used to create this bundle.",
            "",
            "The encoder is the only quantized component in this bundle. VQ quantizer and decode path weights are saved in float form.",
        ]
    )
    (bundle_dir / "README.md").write_text(readme + "\n")

    print("[6/6] Bundle complete")
    print(bundle_dir)


if __name__ == "__main__":
    main()
