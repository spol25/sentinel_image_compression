import argparse
import json
from collections import Counter
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.iwildcam2022 import (
    export_crop,
    load_crop_index,
    manifest_from_paths,
    sample_calibration_eval_candidates,
    select_unique_image_candidates,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Create calibration/eval crop sets and manifests from an iWildCam 2022 crop index.")
    parser.add_argument("--crop-index", required=True, help="Path to crop_index.json produced by build_crop_index.py.")
    parser.add_argument(
        "--output-root",
        default="/Volumes/Media/iwildcam2022_ptq",
        help="Root directory for exported crops and manifests. External drive paths are recommended.",
    )
    parser.add_argument("--calibration-count", type=int, default=300, help="Number of calibration crops to export.")
    parser.add_argument("--eval-count", type=int, default=50, help="Number of eval crops to export.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--max-crops-per-image",
        type=int,
        default=1,
        help="Maximum number of object crops to keep per source image before sampling.",
    )
    parser.add_argument(
        "--image-format",
        choices=("jpg", "png"),
        default="jpg",
        help="Image format used for exported crops.",
    )
    return parser.parse_args()


def crop_filename(candidate, index: int, suffix: str):
    return f"{index:04d}_{candidate.image_id}_det{candidate.detection_index:02d}.{suffix}"


def export_split(split_name: str, candidates, output_root: Path, image_format: str):
    split_dir = output_root / "crops" / split_name
    manifest_paths = []
    records = []
    for index, candidate in enumerate(candidates):
        output_path = split_dir / crop_filename(candidate, index, image_format)
        export_crop(candidate, output_path)
        manifest_paths.append(output_path)
        records.append(
            {
                "crop_path": str(output_path),
                "image_id": candidate.image_id,
                "source_image": candidate.image_path,
                "source_split": candidate.source_split,
                "category_id": candidate.category_id,
                "location": candidate.location,
                "seq_id": candidate.seq_id,
                "detection_conf": candidate.detection_conf,
                "bbox_xyxy_pixels": candidate.bbox_xyxy_pixels,
            }
        )
    return manifest_paths, records


def main():
    args = parse_args()
    crop_index_path = Path(args.crop_index).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = load_crop_index(crop_index_path)
    unique_candidates = select_unique_image_candidates(
        candidates,
        max_crops_per_image=args.max_crops_per_image,
    )
    calibration_candidates, eval_candidates = sample_calibration_eval_candidates(
        unique_candidates,
        calibration_count=args.calibration_count,
        eval_count=args.eval_count,
        seed=args.seed,
    )

    calibration_paths, calibration_records = export_split(
        "calibration", calibration_candidates, output_root, args.image_format
    )
    eval_paths, eval_records = export_split(
        "eval", eval_candidates, output_root, args.image_format
    )

    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    image_manifest_path = manifests_dir / "image_manifest.json"
    calibration_manifest_path = manifests_dir / "calibration_manifest.json"
    eval_manifest_path = manifests_dir / "eval_manifest.json"
    selection_summary_path = manifests_dir / "selection_summary.json"
    crop_records_path = manifests_dir / "crop_records.json"

    all_paths = calibration_paths + eval_paths
    image_manifest_path.write_text(json.dumps(manifest_from_paths(all_paths), indent=2))
    calibration_manifest_path.write_text(
        json.dumps(manifest_from_paths(calibration_paths, source_manifest=image_manifest_path, split="calibration"), indent=2)
    )
    eval_manifest_path.write_text(
        json.dumps(manifest_from_paths(eval_paths, source_manifest=image_manifest_path, split="eval"), indent=2)
    )

    crop_records = {
        "calibration": calibration_records,
        "eval": eval_records,
    }
    crop_records_path.write_text(json.dumps(crop_records, indent=2))

    category_counts = Counter(record["category_id"] for record in calibration_records + eval_records if record["category_id"] is not None)
    location_counts = Counter(record["location"] for record in calibration_records + eval_records if record["location"] is not None)
    summary = {
        "crop_index": str(crop_index_path),
        "output_root": str(output_root),
        "calibration_count": len(calibration_records),
        "eval_count": len(eval_records),
        "max_crops_per_image": args.max_crops_per_image,
        "seed": args.seed,
        "num_unique_categories": len(category_counts),
        "num_unique_locations": len(location_counts),
        "top_categories": category_counts.most_common(20),
        "top_locations": location_counts.most_common(20),
    }
    selection_summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved calibration crops to {output_root / 'crops' / 'calibration'}")
    print(f"Saved eval crops to {output_root / 'crops' / 'eval'}")
    print(f"Saved manifests to {manifests_dir}")


if __name__ == "__main__":
    main()
