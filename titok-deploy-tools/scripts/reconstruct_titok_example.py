import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.decode import decode_token_ids, load_token_json
from titok_deploy_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.utils import (
    load_image,
    resolve_input_path,
    resolve_named_output,
    resolve_output_dir,
    save_reconstruction,
    select_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct an image with a pretrained TiTok tokenizer.")
    parser.add_argument("--titok-root", required=True, help="Path to a separate 1d-tokenizer checkout.")
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_l32_imagenet",
        help="Hugging Face repo containing the pretrained TiTok tokenizer.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Input image path. Defaults to a sample image inside --titok-root/assets.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where all generated outputs will be written.",
    )
    parser.add_argument(
        "--output",
        default="titok_l32_reconstruction.png",
        help="Filename for the reconstructed image inside --output-dir.",
    )
    parser.add_argument(
        "--tokens-input",
        default=None,
        help="Optional path to a JSON file with stored TiTok tokens for decode-only mode.",
    )
    parser.add_argument(
        "--tokens-output",
        default="titok_l32_tokens.json",
        help="Filename for serialized tokens and metadata inside --output-dir.",
    )
    return parser.parse_args()


def encode_to_tokens(tokenizer, image: torch.Tensor, device: str) -> torch.Tensor:
    if tokenizer.quantize_mode == "vq":
        encoded = tokenizer.encode(image.to(device))[1]["min_encoding_indices"]
        return encoded
    if tokenizer.quantize_mode == "vae":
        posterior = tokenizer.encode(image.to(device))[1]
        return posterior.sample()
    raise NotImplementedError(f"Unsupported quantize_mode: {tokenizer.quantize_mode}")


def save_tokens(tokens: torch.Tensor, path: Path, repo_id: str, device: str):
    payload = {
        "repo_id": repo_id,
        "device": device,
        "shape": list(tokens.shape),
        "tokens": tokens.detach().to("cpu").tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    titok_root = add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
    output_path = resolve_named_output(output_dir, args.output)
    tokens_output_path = resolve_named_output(output_dir, args.tokens_output)
    if args.image is None:
        image_path = titok_root / "assets" / "ILSVRC2012_val_00010240.png"
    else:
        image_path = resolve_input_path(args.image)

    device = select_device()
    print(f"[1/5] Loading pretrained tokenizer from {args.repo_id}")
    tokenizer = TiTok.from_pretrained(args.repo_id)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    tokenizer = tokenizer.to(device)

    if args.tokens_input is None:
        image_size = int(tokenizer.config.dataset.preprocessing.crop_size)
        print(f"[2/5] Loading and resizing {image_path} to {image_size}x{image_size}")
        image = load_image(image_path, image_size)

        print("[3/5] Encoding image into TiTok latent tokens")
        tokens = encode_to_tokens(tokenizer, image, device)
        print(f"Encoded token shape: {tuple(tokens.shape)}")

        print("[4/5] Decoding tokens back into pixel space")
        reconstruction = decode_token_ids(tokenizer, tokens, device)

        print(f"[5/5] Saving outputs to {output_path} and {tokens_output_path}")
        save_reconstruction(reconstruction, output_path)
        save_tokens(tokens, tokens_output_path, args.repo_id, device)

        print("[extra] Loading stored tokens and decoding again to confirm token-only reconstruction")
        stored_tokens = load_token_json(tokens_output_path)
        reconstructed_from_stored_tokens = decode_token_ids(tokenizer, stored_tokens, device)
        decode_only_output_path = output_path.with_name(f"{output_path.stem}_from_tokens{output_path.suffix}")
        save_reconstruction(reconstructed_from_stored_tokens, decode_only_output_path)
    else:
        tokens_input_path = resolve_input_path(args.tokens_input, output_dir)
        print(f"[2/5] Loading stored tokens from {tokens_input_path}")
        stored_tokens = load_token_json(tokens_input_path)
        print(f"[3/5] Decoding stored tokens with shape {tuple(stored_tokens.shape)}")
        reconstruction = decode_token_ids(tokenizer, stored_tokens, device)
        print(f"[4/5] Saving decode-only reconstruction to {output_path}")
        save_reconstruction(reconstruction, output_path)
        print("[5/5] Decode-only run complete")

    print("Reconstruction complete.")


if __name__ == "__main__":
    main()
