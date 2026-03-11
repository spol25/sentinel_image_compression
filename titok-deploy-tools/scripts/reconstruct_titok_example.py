import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.titok_env import add_titok_root_to_path


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


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(image_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    return image_tensor


def encode_to_tokens(tokenizer, image: torch.Tensor, device: str) -> torch.Tensor:
    if tokenizer.quantize_mode == "vq":
        encoded = tokenizer.encode(image.to(device))[1]["min_encoding_indices"]
        return encoded
    if tokenizer.quantize_mode == "vae":
        posterior = tokenizer.encode(image.to(device))[1]
        return posterior.sample()
    raise NotImplementedError(f"Unsupported quantize_mode: {tokenizer.quantize_mode}")


def decode_from_tokens(tokenizer, tokens: torch.Tensor, device: str) -> torch.Tensor:
    return tokenizer.decode_tokens(tokens.to(device))


def save_tokens(tokens: torch.Tensor, path: Path, repo_id: str, device: str):
    payload = {
        "repo_id": repo_id,
        "device": device,
        "shape": list(tokens.shape),
        "tokens": tokens.detach().to("cpu").tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_tokens(path: Path) -> torch.Tensor:
    payload = json.loads(path.read_text())
    return torch.tensor(payload["tokens"], dtype=torch.long)


def save_reconstruction(image_tensor: torch.Tensor, path: Path):
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
    image_np = (
        image_tensor[0]
        .permute(1, 2, 0)
        .detach()
        .to("cpu", dtype=torch.float32)
        .numpy()
    )
    image_np = (image_np * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_np).save(path)


def main():
    args = parse_args()
    titok_root = add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(args.output).name
    tokens_output_path = output_dir / Path(args.tokens_output).name
    if args.image is None:
        image_path = titok_root / "assets" / "ILSVRC2012_val_00010240.png"
    else:
        image_path = Path(args.image)

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
        reconstruction = decode_from_tokens(tokenizer, tokens, device)

        print(f"[5/5] Saving outputs to {output_path} and {tokens_output_path}")
        save_reconstruction(reconstruction, output_path)
        save_tokens(tokens, tokens_output_path, args.repo_id, device)

        print("[extra] Loading stored tokens and decoding again to confirm token-only reconstruction")
        stored_tokens = load_tokens(tokens_output_path)
        reconstructed_from_stored_tokens = decode_from_tokens(tokenizer, stored_tokens, device)
        decode_only_output_path = output_path.with_name(f"{output_path.stem}_from_tokens{output_path.suffix}")
        save_reconstruction(reconstructed_from_stored_tokens, decode_only_output_path)
    else:
        tokens_input_path = Path(args.tokens_input)
        if not tokens_input_path.is_absolute():
            tokens_input_path = output_dir / tokens_input_path
        print(f"[2/5] Loading stored tokens from {tokens_input_path}")
        stored_tokens = load_tokens(tokens_input_path)
        print(f"[3/5] Decoding stored tokens with shape {tuple(stored_tokens.shape)}")
        reconstruction = decode_from_tokens(tokenizer, stored_tokens, device)
        print(f"[4/5] Saving decode-only reconstruction to {output_path}")
        save_reconstruction(reconstruction, output_path)
        print("[5/5] Decode-only run complete")

    print("Reconstruction complete.")


if __name__ == "__main__":
    main()
