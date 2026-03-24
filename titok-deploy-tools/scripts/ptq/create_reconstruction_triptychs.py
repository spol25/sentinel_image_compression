import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.decode import decode_token_ids
from titok_deploy_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.utils import load_image, resolve_input_path, resolve_output_dir


LABELS = ("original", "float baseline", "ptq encoder")
HEADER_HEIGHT = 28
GAP = 6
BACKGROUND = (245, 245, 245)
TEXT = (20, 20, 20)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create stitched original/float/PTQ reconstruction panels from token JSON files."
    )
    parser.add_argument("--titok-root", required=True, help="Path to a separate 1d-tokenizer checkout.")
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_s128_imagenet",
        help="Hugging Face repo for the pretrained TiTok-S-128 tokenizer.",
    )
    parser.add_argument("--float-tokens", required=True, help="Float baseline token JSON path.")
    parser.add_argument("--ptq-tokens", required=True, help="PTQ token JSON path.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where stitched preview panels and a manifest will be written.",
    )
    return parser.parse_args()


def load_payload_records(path: Path):
    payload = json.loads(path.read_text())
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Expected token payload with records in {path}")
    return payload, records


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
    image_np = (
        image_tensor[0]
        .permute(1, 2, 0)
        .detach()
        .to("cpu", dtype=torch.float32)
        .numpy()
    )
    return Image.fromarray((image_np * 255.0).astype("uint8"))


def build_triptych(original: Image.Image, float_recon: Image.Image, ptq_recon: Image.Image) -> Image.Image:
    width, height = original.size
    panel_width = width * 3 + GAP * 4
    panel_height = height + HEADER_HEIGHT + GAP * 2
    canvas = Image.new("RGB", (panel_width, panel_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (label, image) in enumerate(zip(LABELS, (original, float_recon, ptq_recon))):
        x = GAP + index * (width + GAP)
        canvas.paste(image, (x, HEADER_HEIGHT + GAP))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (width - text_width) // 2
        draw.text((text_x, 8), label, fill=TEXT, font=font)

    return canvas


def main():
    args = parse_args()
    add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    float_tokens_path = resolve_input_path(args.float_tokens, REPO_ROOT)
    ptq_tokens_path = resolve_input_path(args.ptq_tokens, REPO_ROOT)
    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)

    float_payload, float_records = load_payload_records(float_tokens_path)
    _, ptq_records = load_payload_records(ptq_tokens_path)
    if len(float_records) != len(ptq_records):
        raise ValueError("Float and PTQ token files have different record counts.")

    image_size = int(float_payload.get("image_size") or 256)
    tokenizer = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
    tokenizer.requires_grad_(False)

    manifest = {
        "float_tokens": str(float_tokens_path),
        "ptq_tokens": str(ptq_tokens_path),
        "image_size": image_size,
        "num_images": len(float_records),
        "panels": [],
    }

    for index, (float_record, ptq_record) in enumerate(zip(float_records, ptq_records)):
        image_path = Path(float_record["image"])
        if Path(ptq_record["image"]) != image_path:
            raise ValueError(f"Mismatched images at index {index}: {image_path} vs {ptq_record['image']}")

        original = tensor_to_pil(load_image(image_path, image_size))
        float_tokens = torch.tensor(float_record["tokens"], dtype=torch.long)
        ptq_tokens = torch.tensor(ptq_record["tokens"], dtype=torch.long)
        float_recon = tensor_to_pil(decode_token_ids(tokenizer, float_tokens, "cpu"))
        ptq_recon = tensor_to_pil(decode_token_ids(tokenizer, ptq_tokens, "cpu"))

        panel_name = f"{index:03d}_{image_path.stem}.png"
        panel_path = output_dir / panel_name
        build_triptych(original, float_recon, ptq_recon).save(panel_path)
        manifest["panels"].append(
            {
                "image": str(image_path),
                "panel": str(panel_path),
            }
        )

    (output_dir / "triptych_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(float_records)} stitched panels to {output_dir}")


if __name__ == "__main__":
    main()
