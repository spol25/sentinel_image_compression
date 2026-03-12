import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.utils import resolve_named_output, resolve_output_dir


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a calibration manifest from a directory of images.")
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing representative calibration images.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ptq",
        help="Directory where the manifest and metadata will be written.",
    )
    parser.add_argument(
        "--manifest-name",
        default="calibration_manifest.json",
        help="Filename for the generated calibration manifest.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to include.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
    manifest_path = resolve_named_output(output_dir, args.manifest_name)
    metadata_path = output_dir / "calibration_manifest_metadata.json"

    image_paths = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )
    if args.limit is not None:
        image_paths = image_paths[:args.limit]

    if not image_paths:
        raise SystemExit(f"No calibration images found in {image_dir}")

    manifest = {
        "image_dir": str(image_dir),
        "count": len(image_paths),
        "images": [str(path) for path in image_paths],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    metadata = {
        "manifest_path": str(manifest_path),
        "count": len(image_paths),
        "extensions": sorted(VALID_SUFFIXES),
        "limit": args.limit,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved calibration manifest with {len(image_paths)} image(s) to {manifest_path}")


if __name__ == "__main__":
    main()
