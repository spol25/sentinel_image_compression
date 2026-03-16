import argparse
import json
import random
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.ptq import load_manifest_records
from titok_deploy_tools.utils import resolve_input_path, resolve_named_output, resolve_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Split a generic image manifest into calibration and eval manifests.")
    parser.add_argument(
        "--manifest",
        default="outputs/ptq/image_manifest.json",
        help="Path to the source image manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ptq",
        help="Directory where the calibration and eval manifests will be written.",
    )
    parser.add_argument(
        "--calibration-name",
        default="calibration_manifest.json",
        help="Filename for the calibration manifest.",
    )
    parser.add_argument(
        "--eval-name",
        default="eval_manifest.json",
        help="Filename for the eval manifest.",
    )
    parser.add_argument(
        "--eval-count",
        type=int,
        default=32,
        help="Number of images to reserve for eval.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the manifest before splitting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --shuffle is enabled.",
    )
    return parser.parse_args()


def write_manifest(path: Path, source_manifest: Path, image_paths: list[Path], split: str):
    payload = {
        "source_manifest": str(source_manifest),
        "split": split,
        "count": len(image_paths),
        "images": [str(path) for path in image_paths],
    }
    path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    manifest_path = resolve_input_path(args.manifest, REPO_ROOT)
    image_paths = load_manifest_records(manifest_path)
    if len(image_paths) <= args.eval_count:
        raise SystemExit("Need more images than --eval-count to create non-overlapping calibration and eval manifests.")

    if args.shuffle:
        rng = random.Random(args.seed)
        image_paths = list(image_paths)
        rng.shuffle(image_paths)

    eval_paths = image_paths[: args.eval_count]
    calibration_paths = image_paths[args.eval_count:]

    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
    calibration_path = resolve_named_output(output_dir, args.calibration_name)
    eval_path = resolve_named_output(output_dir, args.eval_name)
    metadata_path = output_dir / "split_manifest_metadata.json"

    write_manifest(calibration_path, manifest_path, calibration_paths, "calibration")
    write_manifest(eval_path, manifest_path, eval_paths, "eval")

    metadata = {
        "source_manifest": str(manifest_path),
        "num_input_images": len(image_paths),
        "num_calibration_images": len(calibration_paths),
        "num_eval_images": len(eval_paths),
        "shuffle": args.shuffle,
        "seed": args.seed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved calibration manifest with {len(calibration_paths)} image(s) to {calibration_path}")
    print(f"Saved eval manifest with {len(eval_paths)} image(s) to {eval_path}")


if __name__ == "__main__":
    main()
