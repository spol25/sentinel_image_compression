import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.utils import load_image, resolve_input_path, select_device
from titok_deploy_tools.wrappers import TiTokTokenEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Validate TiTok-S-128 token wrapper against TiTok.encode().")
    parser.add_argument("--titok-root", required=True, help="Path to a separate 1d-tokenizer checkout.")
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_s128_imagenet",
        help="Hugging Face repo for the pretrained TiTok-S-128 tokenizer.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Image paths to validate against. Defaults to two sample images inside --titok-root/assets.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    titok_root = add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    device = select_device()
    if args.images is None:
        image_paths = [
            titok_root / "assets" / "ILSVRC2012_val_00008636.png",
            titok_root / "assets" / "ILSVRC2012_val_00010240.png",
        ]
    else:
        image_paths = [resolve_input_path(image_arg) for image_arg in args.images]

    print(f"[1/4] Loading TiTok model from {args.repo_id}")
    titok = TiTok.from_pretrained(args.repo_id)
    titok.eval()
    titok.requires_grad_(False)
    titok = titok.to(device)

    print("[2/4] Building minimal token-only wrapper")
    wrapper = TiTokTokenEncoder(titok)
    wrapper.eval()
    wrapper.requires_grad_(False)
    wrapper = wrapper.to(device)

    image_size = int(titok.config.dataset.preprocessing.crop_size)
    print(f"[3/4] Running eager validation on {len(image_paths)} image(s) at {image_size}x{image_size}")
    all_match = True
    for image_path in image_paths:
        image = load_image(image_path, image_size).to(device)

        reference_tokens = titok.encode(image)[1]["min_encoding_indices"].reshape(image.shape[0], -1)
        wrapper_tokens = wrapper(image)

        matches = torch.equal(reference_tokens, wrapper_tokens)
        all_match = all_match and matches
        print(
            f"{image_path}: match={matches} "
            f"shape={tuple(wrapper_tokens.shape)} dtype={wrapper_tokens.dtype}"
        )

        if not matches:
            max_abs_diff = (reference_tokens.to(torch.int64) - wrapper_tokens.to(torch.int64)).abs().max().item()
            print(f"  max token difference: {max_abs_diff}")

    print("[4/4] Validation result")
    if not all_match:
        raise SystemExit("Wrapper tokens do not match TiTok.encode() output.")
    print("All wrapper outputs exactly match TiTok.encode().")


if __name__ == "__main__":
    main()
