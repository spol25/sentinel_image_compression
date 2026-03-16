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

from titok_deploy_tools.snapshot_serengeti import (
    AZURE_BASE_URL,
    build_remote_image_url,
    choose_best_box_per_image,
    download_image,
    export_crop,
    load_candidate_index,
    manifest_payload,
    select_representative_split,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Selectively download and crop a PTQ dataset from Snapshot Serengeti.")
    parser.add_argument("--candidate-index", required=True, help="Path to the candidate index JSON.")
    parser.add_argument(
        "--output-root",
        default="/Volumes/Media/snapshot_serengeti_ptq",
        help="Root directory for downloaded images, cropped images, and manifests.",
    )
    parser.add_argument("--calibration-count", type=int, default=300, help="Number of calibration crops.")
    parser.add_argument("--eval-count", type=int, default=50, help="Number of eval crops.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--base-url",
        default=AZURE_BASE_URL,
        help="Base URL for selective image downloads from cloud storage.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume the selected source images are already present under output-root/downloads.",
    )
    parser.add_argument(
        "--image-format",
        choices=("jpg", "png"),
        default="jpg",
        help="Format for exported crop images.",
    )
    return parser.parse_args()


def crop_filename(candidate, index: int, suffix: str):
    return f"{index:04d}_{Path(candidate.file_name).stem}.{suffix}"


def export_split(split_name: str, candidates, output_root: Path, image_format: str, base_url: str, skip_download: bool):
    downloads_dir = output_root / "downloads"
    crops_dir = output_root / "crops" / split_name
    manifest_paths = []
    records = []
    for index, candidate in enumerate(candidates):
        source_path = downloads_dir / candidate.file_name
        if not skip_download:
            download_image(candidate.file_name, source_path, base_url=base_url)
        elif not source_path.exists():
            raise FileNotFoundError(f"Expected downloaded source image at {source_path}")

        crop_path = crops_dir / crop_filename(candidate, index, image_format)
        export_crop(source_path, candidate.bbox_xyxy_pixels, crop_path)
        manifest_paths.append(crop_path)
        records.append(
            {
                "crop_path": str(crop_path),
                "downloaded_source": str(source_path),
                "remote_url": build_remote_image_url(candidate.file_name, base_url=base_url),
                "file_name": candidate.file_name,
                "category_id": candidate.category_id,
                "category_name": candidate.category_name,
                "location": candidate.location,
                "seq_id": candidate.seq_id,
                "season": candidate.season,
                "bbox_xyxy_pixels": candidate.bbox_xyxy_pixels,
            }
        )
    return manifest_paths, records


def main():
    args = parse_args()
    candidate_index_path = Path(args.candidate_index).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = load_candidate_index(candidate_index_path)
    unique_candidates = choose_best_box_per_image(candidates)
    calibration_candidates, eval_candidates = select_representative_split(
        unique_candidates,
        calibration_count=args.calibration_count,
        eval_count=args.eval_count,
        seed=args.seed,
    )

    calibration_paths, calibration_records = export_split(
        "calibration",
        calibration_candidates,
        output_root,
        args.image_format,
        args.base_url,
        args.skip_download,
    )
    eval_paths, eval_records = export_split(
        "eval",
        eval_candidates,
        output_root,
        args.image_format,
        args.base_url,
        args.skip_download,
    )

    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    image_manifest_path = manifests_dir / "image_manifest.json"
    calibration_manifest_path = manifests_dir / "calibration_manifest.json"
    eval_manifest_path = manifests_dir / "eval_manifest.json"
    crop_records_path = manifests_dir / "crop_records.json"
    selection_summary_path = manifests_dir / "selection_summary.json"

    all_paths = calibration_paths + eval_paths
    image_manifest_path.write_text(json.dumps(manifest_payload(all_paths), indent=2))
    calibration_manifest_path.write_text(
        json.dumps(manifest_payload(calibration_paths, source_manifest=image_manifest_path, split="calibration"), indent=2)
    )
    eval_manifest_path.write_text(
        json.dumps(manifest_payload(eval_paths, source_manifest=image_manifest_path, split="eval"), indent=2)
    )
    crop_records_path.write_text(json.dumps({"calibration": calibration_records, "eval": eval_records}, indent=2))

    category_counts = Counter(record["category_name"] for record in calibration_records + eval_records if record["category_name"])
    location_counts = Counter(record["location"] for record in calibration_records + eval_records if record["location"])
    season_counts = Counter(record["season"] for record in calibration_records + eval_records if record["season"])
    summary = {
        "candidate_index": str(candidate_index_path),
        "output_root": str(output_root),
        "calibration_count": len(calibration_records),
        "eval_count": len(eval_records),
        "seed": args.seed,
        "base_url": args.base_url,
        "num_unique_categories": len(category_counts),
        "num_unique_locations": len(location_counts),
        "num_unique_seasons": len(season_counts),
        "top_categories": category_counts.most_common(25),
        "top_locations": location_counts.most_common(25),
        "top_seasons": season_counts.most_common(10),
    }
    selection_summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved calibration crops to {output_root / 'crops' / 'calibration'}")
    print(f"Saved eval crops to {output_root / 'crops' / 'eval'}")
    print(f"Saved manifests to {manifests_dir}")


if __name__ == "__main__":
    main()
