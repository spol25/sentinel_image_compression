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
from titok_deploy_tools.utils import load_image
from titok_deploy_tools.wrappers import TiTokTokenEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Export the TiTok-S-128 token wrapper with torch.export.")
    parser.add_argument("--titok-root", required=True, help="Path to a separate 1d-tokenizer checkout.")
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_s128_imagenet",
        help="Hugging Face repo for the pretrained TiTok-S-128 tokenizer.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Input image path used as the export example input. Defaults to a sample image in --titok-root/assets.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/export",
        help="Directory where the export artifact and metadata will be written.",
    )
    parser.add_argument(
        "--artifact-name",
        default="titok_s128_token_encoder.pt2",
        help="Filename for the exported torch.export artifact.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    titok_root = add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    if args.image is None:
        image_path = titok_root / "assets" / "ILSVRC2012_val_00010240.png"
    else:
        image_path = Path(args.image)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / Path(args.artifact_name).name
    metadata_path = output_dir / "titok_s128_token_encoder_metadata.json"

    print(f"[1/5] Loading TiTok model from {args.repo_id} on CPU")
    titok = TiTok.from_pretrained(args.repo_id)
    titok.eval()
    titok.requires_grad_(False)
    titok = titok.to("cpu")

    print("[2/5] Building export-oriented token wrapper")
    wrapper = TiTokTokenEncoder(titok)
    wrapper.eval()
    wrapper.requires_grad_(False)
    wrapper = wrapper.to("cpu")

    image_size = int(titok.config.dataset.preprocessing.crop_size)
    example_input = load_image(image_path, image_size).to("cpu")

    print(f"[3/5] Running eager wrapper on example input {image_path}")
    eager_tokens = wrapper(example_input)

    print("[4/5] Exporting wrapper with torch.export and validating output parity")
    exported_program = torch.export.export(wrapper, (example_input,))
    exported_tokens = exported_program.module()(example_input)
    if not torch.equal(eager_tokens, exported_tokens):
        raise SystemExit("Exported wrapper output does not match eager wrapper output.")
    torch.export.save(exported_program, artifact_path)

    print(f"[5/5] Saving export metadata to {metadata_path}")
    metadata = {
        "repo_id": args.repo_id,
        "artifact_path": str(artifact_path),
        "image_size": image_size,
        "input_shape": list(example_input.shape),
        "output_shape": list(eager_tokens.shape),
        "output_dtype": str(eager_tokens.dtype),
        "token_count": eager_tokens.shape[-1],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved export artifact to {artifact_path}")


if __name__ == "__main__":
    main()
