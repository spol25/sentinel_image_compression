import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.utils import load_image, resolve_input_path, resolve_output_dir
from titok_deploy_tools.wrappers import TiTokTokenEncoder


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
    baseline_path = output_dir / "s128_calibration_baseline_tokens.json"
    summary_path = output_dir / "s128_calibration_baseline_summary.json"

    manifest = json.loads(manifest_path.read_text())
    image_paths = [Path(p) for p in manifest["images"]]

    print(f"[1/4] Loading TiTok model from {args.repo_id} on CPU")
    titok = TiTok.from_pretrained(args.repo_id)
    titok.eval()
    titok.requires_grad_(False)
    titok = titok.to("cpu")

    print("[2/4] Building token-only wrapper on CPU")
    wrapper = TiTokTokenEncoder(titok)
    wrapper.eval()
    wrapper.requires_grad_(False)
    wrapper = wrapper.to("cpu")

    image_size = int(titok.config.dataset.preprocessing.crop_size)
    print(f"[3/4] Running wrapper over {len(image_paths)} calibration image(s)")
    records = []
    unique_token_ids = set()
    token_count = None
    for image_path in image_paths:
        image = load_image(image_path, image_size).to("cpu")
        tokens = wrapper(image).to("cpu", dtype=torch.int64)
        token_list = tokens[0].tolist()
        unique_token_ids.update(token_list)
        token_count = len(token_list)
        records.append({
            "image": str(image_path),
            "tokens": token_list,
        })

    print(f"[4/4] Saving baseline outputs to {baseline_path} and {summary_path}")
    baseline_payload = {
        "repo_id": args.repo_id,
        "image_size": image_size,
        "token_shape": [1, token_count],
        "records": records,
    }
    baseline_path.write_text(json.dumps(baseline_payload, indent=2))

    summary = {
        "repo_id": args.repo_id,
        "manifest_path": str(manifest_path),
        "num_images": len(records),
        "token_count_per_image": token_count,
        "num_unique_token_ids": len(unique_token_ids),
        "min_token_id": min(unique_token_ids),
        "max_token_id": max(unique_token_ids),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved calibration baseline for {len(records)} image(s)")


if __name__ == "__main__":
    main()
