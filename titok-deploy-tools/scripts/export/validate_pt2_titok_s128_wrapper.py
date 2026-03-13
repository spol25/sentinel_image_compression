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

from titok_deploy_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.utils import load_image, resolve_input_path, resolve_named_output, resolve_output_dir
from titok_deploy_tools.wrappers import TiTokTokenEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a saved TiTok S-128 .pt2 export artifact.")
    parser.add_argument("--titok-root", required=True, help="Path to a separate 1d-tokenizer checkout.")
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_s128_imagenet",
        help="Hugging Face repo for the pretrained TiTok-S-128 tokenizer.",
    )
    parser.add_argument(
        "--pt2-path",
        default="outputs/export/titok_s128_token_encoder.pt2",
        help="Path to the saved .pt2 export artifact.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Input image path used for validation. Defaults to a sample image in --titok-root/assets.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/test_s128",
        help="Directory where validation outputs will be written.",
    )
    parser.add_argument(
        "--tokens-output",
        default="titok_s128_pt2_tokens.json",
        help="Filename for serialized token output inside --output-dir.",
    )
    return parser.parse_args()


def save_tokens(tokens: torch.Tensor, path: Path, repo_id: str):
    payload = {
        "repo_id": repo_id,
        "shape": list(tokens.shape),
        "tokens": tokens.detach().to("cpu").tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    titok_root = add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    pt2_path = resolve_input_path(args.pt2_path, REPO_ROOT)
    if not pt2_path.exists():
        raise FileNotFoundError(f".pt2 artifact not found: {pt2_path}")

    if args.image is None:
        image_path = titok_root / "assets" / "ILSVRC2012_val_00010240.png"
    else:
        image_path = resolve_input_path(args.image)

    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
    tokens_output_path = resolve_named_output(output_dir, args.tokens_output)

    print(f"[1/5] Loading TiTok model from {args.repo_id} on CPU for eager reference")
    titok = TiTok.from_pretrained(args.repo_id)
    titok.eval()
    titok.requires_grad_(False)
    titok = titok.to("cpu")

    print("[2/5] Building eager token wrapper on CPU")
    wrapper = TiTokTokenEncoder(titok)
    wrapper.eval()
    wrapper.requires_grad_(False)
    wrapper = wrapper.to("cpu")

    image_size = int(titok.config.dataset.preprocessing.crop_size)
    example_input = load_image(image_path, image_size).to("cpu")

    print(f"[3/5] Running eager reference on {image_path}")
    eager_tokens = wrapper(example_input)

    print(f"[4/5] Loading exported program from {pt2_path} and validating parity")
    exported_program = torch.export.load(pt2_path)
    exported_tokens = exported_program.module()(example_input)
    if not torch.equal(eager_tokens, exported_tokens):
        raise SystemExit("Loaded .pt2 program output does not match eager wrapper output.")

    print(f"[5/5] Saving exported-program tokens to {tokens_output_path}")
    save_tokens(exported_tokens, tokens_output_path, args.repo_id)
    print("pt2 validation passed.")


if __name__ == "__main__":
    main()
