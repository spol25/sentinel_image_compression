import argparse
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.decode import decode_token_ids, load_token_json
from titok_deploy_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.utils import resolve_input_path, resolve_named_output, resolve_output_dir, save_reconstruction, select_device


def parse_args():
    parser = argparse.ArgumentParser(description="Validate decode.py using token JSON produced by validate_titok_s128_wrapper.py.")
    parser.add_argument("--titok-root", required=True, help="Path to a separate 1d-tokenizer checkout.")
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_s128_imagenet",
        help="Hugging Face repo for the pretrained TiTok tokenizer.",
    )
    parser.add_argument(
        "--tokens-json",
        default="s128_wrapper_tokens.json",
        help="Token JSON file produced by validate_titok_s128_wrapper.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where generated outputs will be written.",
    )
    parser.add_argument(
        "--output",
        default="validated_decoded_tokens.png",
        help="Filename for the decoded image inside --output-dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    device = select_device()
    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
    output_path = resolve_named_output(output_dir, args.output)
    tokens_path = resolve_input_path(args.tokens_json, output_dir)

    print(f"[1/4] Loading pretrained tokenizer from {args.repo_id}")
    tokenizer = TiTok.from_pretrained(args.repo_id)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    tokenizer = tokenizer.to(device)

    print(f"[2/4] Loading wrapper token JSON from {tokens_path}")
    token_ids = load_token_json(tokens_path)

    print("[3/4] Validating decode.py output against direct TiTok.decode_tokens()")
    decoded = decode_token_ids(tokenizer, token_ids, device)
    reference = tokenizer.decode_tokens(token_ids.to(device))
    decoded_np = (
        torch.clamp(decoded, 0.0, 1.0)[0]
        .permute(1, 2, 0)
        .detach()
        .to("cpu", dtype=torch.float32)
        .numpy()
    )
    reference_np = (
        torch.clamp(reference, 0.0, 1.0)[0]
        .permute(1, 2, 0)
        .detach()
        .to("cpu", dtype=torch.float32)
        .numpy()
    )
    max_abs_diff = np.abs(decoded_np - reference_np).max()
    if max_abs_diff != 0.0:
        raise SystemExit(f"decode.py output differs from direct decode, max_abs_diff={max_abs_diff:.8f}")

    print(f"[4/4] Saving validated decoded image to {output_path}")
    save_reconstruction(decoded, output_path)
    print("decode.py validation passed.")


if __name__ == "__main__":
    main()
