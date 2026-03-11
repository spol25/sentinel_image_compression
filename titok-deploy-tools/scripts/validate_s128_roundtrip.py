import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.decode import decode_token_ids
from titok_deploy_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.utils import load_image, save_reconstruction, select_device
from titok_deploy_tools.wrappers import TiTokTokenEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Validate S-128 wrapper output can be decoded directly.")
    parser.add_argument("--titok-root", required=True, help="Path to a separate 1d-tokenizer checkout.")
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_s128_imagenet",
        help="Hugging Face repo for the pretrained TiTok-S-128 tokenizer.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Input image path. Defaults to a sample image inside --titok-root/assets.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where generated outputs will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    titok_root = add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    device = select_device()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image is None:
        image_path = titok_root / "assets" / "ILSVRC2012_val_00010240.png"
    else:
        image_path = Path(args.image)

    print(f"[1/5] Loading TiTok model from {args.repo_id}")
    tokenizer = TiTok.from_pretrained(args.repo_id)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    tokenizer = tokenizer.to(device)

    print("[2/5] Building token-only wrapper")
    wrapper = TiTokTokenEncoder(tokenizer)
    wrapper.eval()
    wrapper.requires_grad_(False)
    wrapper = wrapper.to(device)

    image_size = int(tokenizer.config.dataset.preprocessing.crop_size)
    print(f"[3/5] Encoding {image_path} to transmitted token IDs")
    image = load_image(image_path, image_size).to(device)
    wrapper_tokens = wrapper(image)
    print(f"Wrapper token shape: {tuple(wrapper_tokens.shape)} dtype={wrapper_tokens.dtype}")

    print("[4/5] Decoding transmitted token IDs directly")
    decoded = decode_token_ids(tokenizer, wrapper_tokens, device)
    decode_path = output_dir / "s128_roundtrip_decoded.png"
    save_reconstruction(decoded, decode_path)

    print("[5/5] Comparing direct decode with TiTok.decode_tokens() reference")
    reference = tokenizer.decode_tokens(wrapper_tokens.unsqueeze(1))
    reference_np = (
        torch.clamp(reference, 0.0, 1.0)[0]
        .permute(1, 2, 0)
        .detach()
        .to("cpu", dtype=torch.float32)
        .numpy()
    )
    decoded_np = (
        torch.clamp(decoded, 0.0, 1.0)[0]
        .permute(1, 2, 0)
        .detach()
        .to("cpu", dtype=torch.float32)
        .numpy()
    )
    max_abs_diff = np.abs(reference_np - decoded_np).max()
    print(f"Max abs diff vs reference decode: {max_abs_diff:.8f}")
    if max_abs_diff != 0.0:
        raise SystemExit("Round-trip decode does not match TiTok.decode_tokens() reference.")
    print(f"Saved round-trip reconstruction to {decode_path}")


if __name__ == "__main__":
    main()
