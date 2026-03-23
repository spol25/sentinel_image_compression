import argparse
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.ptq import (
    build_encoder_quantizer_split,
    load_manifest_records,
    save_token_records,
    summarize_token_records,
)
from titok_deploy_tools.utils import load_image, resolve_input_path, resolve_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run the S-128 wrapper over a calibration manifest and save baseline outputs.")
    parser.add_argument("--titok-root", required=True, help="Path to a separate 1d-tokenizer checkout.")
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_s128_imagenet",
        help="Hugging Face repo for the pretrained TiTok-S-128 tokenizer.",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/ptq/calibration_manifest.json",
        help="Path to a calibration manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ptq",
        help="Directory where baseline token outputs and metrics will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    manifest_path = resolve_input_path(args.manifest, REPO_ROOT)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Calibration manifest not found: {manifest_path}")

    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
    baseline_path = output_dir / "s128_float_baseline_tokens.json"

    image_paths = load_manifest_records(manifest_path)

    print(f"[1/4] Loading TiTok model from {args.repo_id} on CPU")
    titok = TiTok.from_pretrained(args.repo_id)
    titok.eval()
    titok.requires_grad_(False)
    titok = titok.to("cpu")

    print("[2/4] Building token-only wrapper on CPU")
    _, _, wrapper = build_encoder_quantizer_split(titok)
    wrapper.eval()
    wrapper.requires_grad_(False)
    wrapper = wrapper.to("cpu")

    image_size = int(titok.config.dataset.preprocessing.crop_size)
    print(f"[3/4] Running wrapper over {len(image_paths)} calibration image(s)")
    records = []
    token_shape = None
    for image_path in image_paths:
        image = load_image(image_path, image_size).to("cpu")
        tokens = wrapper(image).to("cpu", dtype=torch.int64)
        token_list = tokens[0].tolist()
        token_shape = list(tokens.shape)
        records.append({
            "image": str(image_path),
            "tokens": token_list,
        })

    summary = summarize_token_records(records) | {
        "repo_id": args.repo_id,
        "manifest_path": str(manifest_path),
        "source": "float_wrapper_baseline",
    }

    print(f"[4/4] Saving baseline outputs to {baseline_path}")
    save_token_records(
        baseline_path,
        records,
        repo_id=args.repo_id,
        image_size=image_size,
        token_shape=token_shape,
        metadata={
            "manifest_path": str(manifest_path),
            "source": "float_wrapper_baseline",
        },
        summary=summary,
    )
    print(f"Saved calibration baseline for {len(records)} image(s)")


if __name__ == "__main__":
    main()
