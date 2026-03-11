import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.decode import decode_token_ids, load_token_json
from titok_deploy_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.utils import save_reconstruction, select_device


def parse_args():
    parser = argparse.ArgumentParser(description="Decode transmitted TiTok token IDs into an image.")
    parser.add_argument("--titok-root", required=True, help="Path to a separate 1d-tokenizer checkout.")
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_s128_imagenet",
        help="Hugging Face repo for the pretrained TiTok tokenizer.",
    )
    parser.add_argument("--tokens-json", required=True, help="Path to a JSON file containing transmitted token IDs.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where generated outputs will be written.",
    )
    parser.add_argument(
        "--output",
        default="decoded_from_tokens.png",
        help="Filename for the decoded image inside --output-dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    device = select_device()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(args.output).name

    tokens_path = Path(args.tokens_json)
    if not tokens_path.is_absolute():
        tokens_path = output_dir / tokens_path

    print(f"[1/4] Loading pretrained tokenizer from {args.repo_id}")
    tokenizer = TiTok.from_pretrained(args.repo_id)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    tokenizer = tokenizer.to(device)

    print(f"[2/4] Loading transmitted token IDs from {tokens_path}")
    token_ids = load_token_json(tokens_path)
    print(f"Loaded token tensor shape: {tuple(token_ids.shape)} dtype={token_ids.dtype}")

    print("[3/4] Decoding token IDs into pixel space")
    reconstruction = decode_token_ids(tokenizer, token_ids, device)

    print(f"[4/4] Saving decoded image to {output_path}")
    save_reconstruction(reconstruction, output_path)


if __name__ == "__main__":
    main()
