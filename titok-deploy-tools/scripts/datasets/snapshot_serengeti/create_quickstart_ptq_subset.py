import argparse
import json
import random
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
    choose_best_quick_box_per_image,
    download_image,
    export_crop_from_xywh,
    iter_quick_animal_candidates,
    manifest_payload,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Quick bbox-only Snapshot Serengeti PTQ subset builder.")
    parser.add_argument("--bboxes-json", required=True, help="Path to Snapshot Serengeti bounding-box JSON.")
    parser.add_argument(
        "--output-root",
        default="/Volumes/Media/snapshot_serengeti_ptq_quickstart",
        help="Root directory for downloaded images, crops, and manifests.",
    )
    parser.add_argument("--calibration-count", type=int, default=300, help="Number of calibration crops.")
    parser.add_argument("--eval-count", type=int, default=50, help="Number of eval crops.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--base-url",
        default=AZURE_BASE_URL,
        help="Base URL for selective image downloads.",
    )
    parser.add_argument(
        "--margin-fraction",
        type=float,
        default=0.05,
        help="Fractional crop margin applied after download.",
    )
    return parser.parse_args()


def sample_diverse_subset(candidates, subset_count: int, seed: int):
    rng = random.Random(seed)
    pool = list(candidates)
    rng.shuffle(pool)

    selected = []
    season_counts = Counter()
    location_counts = Counter()
    camera_counts = Counter()
    while len(selected) < subset_count and pool:
        best_index = None
        best_score = None
        for index, candidate in enumerate(pool):
            score = 0.0
            score += min(candidate.bbox_area_pixels / 50000.0, 4.0)
            if candidate.season is not None:
                score += 3.0 / (1 + season_counts[candidate.season])
            if candidate.location is not None:
                score += 4.0 / (1 + location_counts[candidate.location])
            if candidate.camera_id is not None:
                score += 2.0 / (1 + camera_counts[candidate.camera_id])
            score += rng.random() * 1e-6
            if best_score is None or score > best_score:
                best_score = score
                best_index = index
        chosen = pool.pop(best_index)
        selected.append(chosen)
        if chosen.season is not None:
            season_counts[chosen.season] += 1
        if chosen.location is not None:
            location_counts[chosen.location] += 1
        if chosen.camera_id is not None:
            camera_counts[chosen.camera_id] += 1
    if len(selected) != subset_count:
        raise ValueError(f"Unable to sample {subset_count} unique images from pool of {len(candidates)}")
    return selected


def export_split(split_name: str, candidates, target_count: int, output_root: Path, base_url: str, margin_fraction: float):
    downloads_dir = output_root / "downloads"
    crops_dir = output_root / "crops" / split_name
    manifest_paths = []
    records = []
    for candidate in candidates:
        if len(records) >= target_count:
            break
        source_path = downloads_dir / candidate.file_name
        try:
            download_image(candidate.file_name, source_path, base_url=base_url)
            crop_path = crops_dir / f"{len(records):04d}_{Path(candidate.file_name).stem}.jpg"
            export_crop_from_xywh(source_path, candidate.bbox_xywh_pixels, crop_path, margin_fraction=margin_fraction)
            manifest_paths.append(crop_path)
            records.append(
                {
                    "crop_path": str(crop_path),
                    "downloaded_source": str(source_path),
                    "remote_url": build_remote_image_url(candidate.file_name, base_url=base_url),
                    "file_name": candidate.file_name,
                    "season": candidate.season,
                    "location": candidate.location,
                    "camera_id": candidate.camera_id,
                    "bbox_xywh_pixels": candidate.bbox_xywh_pixels,
                }
            )
        except Exception:
            if source_path.exists():
                source_path.unlink()
            continue
    if len(records) < target_count:
        raise RuntimeError(f"Only exported {len(records)} {split_name} crops; target was {target_count}.")
    return manifest_paths, records


def main():
    args = parse_args()
    bbox_path = Path(args.bboxes_json).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = list(iter_quick_animal_candidates(bbox_path))
    unique_candidates = choose_best_quick_box_per_image(candidates)
    total_needed = args.calibration_count + args.eval_count
    if len(unique_candidates) < total_needed:
        raise SystemExit(f"Need at least {total_needed} unique animal images, found {len(unique_candidates)}")

    eval_candidates = sample_diverse_subset(unique_candidates, min(len(unique_candidates), args.eval_count * 3), args.seed)
    chosen_files = {candidate.file_name for candidate in eval_candidates}
    remaining = [candidate for candidate in unique_candidates if candidate.file_name not in chosen_files]
    calibration_candidates = sample_diverse_subset(remaining, min(len(remaining), args.calibration_count * 3), args.seed + 1)

    calibration_paths, calibration_records = export_split(
        "calibration",
        calibration_candidates,
        args.calibration_count,
        output_root,
        args.base_url,
        args.margin_fraction,
    )
    eval_paths, eval_records = export_split(
        "eval",
        eval_candidates,
        args.eval_count,
        output_root,
        args.base_url,
        args.margin_fraction,
    )

    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    image_manifest_path = manifests_dir / "image_manifest.json"
    calibration_manifest_path = manifests_dir / "calibration_manifest.json"
    eval_manifest_path = manifests_dir / "eval_manifest.json"
    records_path = manifests_dir / "crop_records.json"
    summary_path = manifests_dir / "selection_summary.json"

    all_paths = calibration_paths + eval_paths
    image_manifest_path.write_text(json.dumps(manifest_payload(all_paths), indent=2))
    calibration_manifest_path.write_text(
        json.dumps(manifest_payload(calibration_paths, source_manifest=image_manifest_path, split="calibration"), indent=2)
    )
    eval_manifest_path.write_text(
        json.dumps(manifest_payload(eval_paths, source_manifest=image_manifest_path, split="eval"), indent=2)
    )
    records_path.write_text(json.dumps({"calibration": calibration_records, "eval": eval_records}, indent=2))

    location_counts = Counter(record["location"] for record in calibration_records + eval_records if record["location"])
    season_counts = Counter(record["season"] for record in calibration_records + eval_records if record["season"])
    camera_counts = Counter(record["camera_id"] for record in calibration_records + eval_records if record["camera_id"])
    summary = {
        "bboxes_json": str(bbox_path),
        "output_root": str(output_root),
        "calibration_count": len(calibration_records),
        "eval_count": len(eval_records),
        "seed": args.seed,
        "num_unique_locations": len(location_counts),
        "num_unique_seasons": len(season_counts),
        "num_unique_cameras": len(camera_counts),
        "top_locations": location_counts.most_common(25),
        "top_seasons": season_counts.most_common(10),
        "top_cameras": camera_counts.most_common(25),
        "selection_mode": "quickstart_bbox_only_animals",
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved calibration crops to {output_root / 'crops' / 'calibration'}")
    print(f"Saved eval crops to {output_root / 'crops' / 'eval'}")
    print(f"Saved manifests to {manifests_dir}")


if __name__ == "__main__":
    main()
