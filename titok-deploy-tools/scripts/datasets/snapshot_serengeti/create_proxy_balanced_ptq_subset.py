import argparse
import json
import random
from collections import Counter
from pathlib import Path
import statistics
import sys

from PIL import Image, ImageStat

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
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a balanced Snapshot Serengeti PTQ subset using bbox-area and brightness proxies."
    )
    parser.add_argument("--candidate-index", required=True, help="Path to the Snapshot Serengeti candidate index JSON.")
    parser.add_argument(
        "--output-root",
        default="/Volumes/Media/snapshot_serengeti_ptq/proxy_balanced",
        help="Root directory for downloads, crops, and manifests.",
    )
    parser.add_argument(
        "--shortlist-count-per-proximity",
        type=int,
        default=1000,
        help="Number of unique source images to shortlist for each of the near/far proximity buckets.",
    )
    parser.add_argument(
        "--calibration-per-bin",
        type=int,
        default=125,
        help="Number of crops per category bin for calibration.",
    )
    parser.add_argument(
        "--eval-per-bin",
        type=int,
        default=25,
        help="Number of crops per category bin for eval.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--base-url",
        default=AZURE_BASE_URL,
        help="Base URL for selective image downloads from cloud storage.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume shortlisted source images already exist under output-root/downloads.",
    )
    parser.add_argument(
        "--image-format",
        choices=("jpg", "png"),
        default="jpg",
        help="Format for exported crop images.",
    )
    parser.add_argument("--image-width", type=int, default=2048, help="Full image width used for bbox-area normalization.")
    parser.add_argument(
        "--image-height", type=int, default=1536, help="Full image height used for bbox-area normalization."
    )
    return parser.parse_args()


def crop_filename(record: dict, index: int, suffix: str) -> str:
    stem = Path(record["file_name"]).stem
    return f"{index:04d}_{record['time_of_day']}_{record['proximity']}_{stem}.{suffix}"


def brightness_stats(image_path: Path) -> dict[str, float]:
    with Image.open(image_path) as image:
        grayscale = image.convert("L")
        stat = ImageStat.Stat(grayscale)
        mean = float(stat.mean[0])
        stddev = float(stat.stddev[0]) if stat.stddev else 0.0
        extrema = grayscale.getextrema()
        histogram = grayscale.histogram()

        hsv = image.convert("HSV")
        saturation = hsv.getchannel("S")
        saturation_stat = ImageStat.Stat(saturation)
        saturation_mean = float(saturation_stat.mean[0])

    total = max(sum(histogram), 1)
    cumulative = 0
    percentile_targets = {
        "brightness_p10": total * 0.10,
        "brightness_p25": total * 0.25,
        "brightness_p50": total * 0.50,
        "brightness_p75": total * 0.75,
        "brightness_p90": total * 0.90,
    }
    percentile_values: dict[str, float] = {}
    remaining = dict(percentile_targets)
    for value, count in enumerate(histogram):
        cumulative += count
        for key, target in list(remaining.items()):
            if cumulative >= target:
                percentile_values[key] = float(value)
                del remaining[key]
        if not remaining:
            break

    return {
        "brightness_mean": mean,
        "brightness_stddev": stddev,
        "brightness_min": float(extrema[0]),
        "brightness_max": float(extrema[1]),
        "saturation_mean": saturation_mean,
        **percentile_values,
    }


def classify_night_score(record: dict) -> float:
    # Lower is more night-like: darker image and lower color saturation.
    luminance_term = (
        0.45 * record["brightness_mean"]
        + 0.35 * record["brightness_p90"]
        + 0.20 * record["brightness_p50"]
    )
    saturation_term = 0.35 * record["saturation_mean"]
    contrast_bonus = 0.10 * record["brightness_stddev"]
    return luminance_term + saturation_term + contrast_bonus


def otsu_threshold(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute Otsu threshold for an empty list.")
    if len(values) == 1:
        return values[0]

    scaled = [max(0, min(255, int(round(value)))) for value in values]
    histogram = [0] * 256
    for value in scaled:
        histogram[value] += 1

    total = len(scaled)
    sum_total = sum(index * count for index, count in enumerate(histogram))
    sum_background = 0.0
    weight_background = 0
    max_variance = -1.0
    threshold = scaled[0]

    for index, count in enumerate(histogram):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += index * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = index
    return float(threshold)


def diversify_records(records: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)

    selected: list[dict] = []
    category_counts = Counter()
    location_counts = Counter()
    season_counts = Counter()
    used_seq_ids = set()

    while len(selected) < count and pool:
        best_index = None
        best_score = None
        for index, record in enumerate(pool):
            score = 0.0
            score += 6.0 / (1 + category_counts[record["category_name"]]) if record["category_name"] else 0.0
            score += 3.0 / (1 + location_counts[record["location"]]) if record["location"] else 0.0
            score += 2.0 / (1 + season_counts[record["season"]]) if record["season"] else 0.0
            if record["seq_id"] and record["seq_id"] not in used_seq_ids:
                score += 2.0
            score += min(record["bbox_area_fraction"] * 10.0, 2.0) if record["proximity"] == "near" else 0.0
            score += min((1.0 - record["bbox_area_fraction"]) * 10.0, 2.0) if record["proximity"] == "far" else 0.0
            score += rng.random() * 1e-6
            if best_score is None or score > best_score:
                best_score = score
                best_index = index

        chosen = pool.pop(best_index)
        selected.append(chosen)
        if chosen["category_name"]:
            category_counts[chosen["category_name"]] += 1
        if chosen["location"]:
            location_counts[chosen["location"]] += 1
        if chosen["season"]:
            season_counts[chosen["season"]] += 1
        if chosen["seq_id"]:
            used_seq_ids.add(chosen["seq_id"])

    if len(selected) != count:
        raise RuntimeError(f"Unable to select {count} records from pool of {len(records)}.")
    return selected


def build_ranked_records(candidates, full_area: int):
    records = []
    for candidate in choose_best_box_per_image(candidates):
        bbox_fraction = float(candidate.bbox_area_pixels) / float(full_area)
        records.append(
            {
                "image_id": candidate.image_id,
                "file_name": candidate.file_name,
                "bbox_xyxy_pixels": candidate.bbox_xyxy_pixels,
                "bbox_area_pixels": float(candidate.bbox_area_pixels),
                "bbox_area_fraction": bbox_fraction,
                "category_id": candidate.category_id,
                "category_name": candidate.category_name,
                "location": candidate.location,
                "seq_id": candidate.seq_id,
                "datetime": candidate.datetime,
                "season": candidate.season,
            }
        )

    return sorted(records, key=lambda record: (record["bbox_area_fraction"], record["bbox_area_pixels"], record["file_name"]))


def download_shortlist(
    pool: list[dict],
    target_count: int,
    downloads_dir: Path,
    base_url: str,
    skip_download: bool,
    exclude_files: set[str] | None = None,
):
    selected = []
    failures = []
    exclude_files = set() if exclude_files is None else set(exclude_files)

    for record in pool:
        if len(selected) >= target_count:
            break
        if record["file_name"] in exclude_files:
            continue
        source_path = downloads_dir / record["file_name"]
        try:
            if skip_download:
                if not source_path.exists():
                    raise FileNotFoundError(f"Expected downloaded source image at {source_path}")
            else:
                download_image(record["file_name"], source_path, base_url=base_url)
            enriched = {
                **record,
                "downloaded_source": str(source_path),
                "remote_url": build_remote_image_url(record["file_name"], base_url=base_url),
            }
            selected.append(enriched)
            exclude_files.add(record["file_name"])
        except Exception as exc:
            failures.append({"file_name": record["file_name"], "error": str(exc), "proximity": record["proximity"]})
            if source_path.exists():
                source_path.unlink()

    if len(selected) < target_count:
        raise RuntimeError(
            f"Only downloaded {len(selected)} records for {pool[0]['proximity'] if pool else 'unknown'}; "
            f"target was {target_count}."
        )
    return selected, failures


def annotate_brightness(records: list[dict]):
    for record in records:
        stats = brightness_stats(Path(record["downloaded_source"]))
        record.update(stats)


def classify_time_of_day(records: list[dict], required_per_class: int, seed: int):
    if len(records) < required_per_class * 2:
        raise RuntimeError(f"Need at least {required_per_class * 2} records in a proximity bucket, found {len(records)}.")

    night_scores = []
    for record in records:
        score = classify_night_score(record)
        record["night_score"] = score
        night_scores.append(score)

    threshold = otsu_threshold(night_scores)

    sorted_records = sorted(records, key=lambda record: (record["night_score"], record["file_name"]))
    tail_pool_size = min(len(sorted_records) // 2, max(required_per_class, required_per_class * 3))
    night_pool = [dict(record, time_of_day="night") for record in sorted_records[:tail_pool_size]]
    day_pool = [dict(record, time_of_day="day") for record in sorted_records[-tail_pool_size:]]

    night_records = diversify_records(night_pool, required_per_class, seed)
    day_records = diversify_records(day_pool, required_per_class, seed + 1000)

    if {record["file_name"] for record in night_records} & {record["file_name"] for record in day_records}:
        raise RuntimeError("Night/day selection overlapped; increase shortlist size.")

    for record in records:
        record["night_score_threshold"] = threshold
        record["brightness_label_from_threshold"] = "day" if record["night_score"] >= threshold else "night"

    rng = random.Random(seed)
    rng.shuffle(night_records)
    rng.shuffle(day_records)
    return night_records, day_records, threshold


def export_split(split_name: str, records: list[dict], output_root: Path, image_format: str):
    crops_dir = output_root / "crops" / split_name
    manifest_paths = []
    exported_records = []
    for index, record in enumerate(records):
        crop_path = crops_dir / crop_filename(record, index, image_format)
        export_crop(Path(record["downloaded_source"]), record["bbox_xyxy_pixels"], crop_path)
        manifest_paths.append(crop_path)
        exported_records.append(
            {
                **record,
                "crop_path": str(crop_path),
            }
        )
    return manifest_paths, exported_records


def split_for_calibration_and_eval(records: list[dict], calibration_per_bin: int, eval_per_bin: int):
    total_needed = calibration_per_bin + eval_per_bin
    if len(records) < total_needed:
        raise RuntimeError(f"Need {total_needed} records for a category bin, found {len(records)}.")
    return records[:calibration_per_bin], records[calibration_per_bin:total_needed]


def main():
    args = parse_args()
    candidate_index_path = Path(args.candidate_index).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    full_area = args.image_width * args.image_height
    required_per_bin = args.calibration_per_bin + args.eval_per_bin

    candidates = load_candidate_index(candidate_index_path)
    ranked_records = build_ranked_records(candidates, full_area)
    if len(ranked_records) < args.shortlist_count_per_proximity * 2:
        raise RuntimeError(
            f"Need at least {args.shortlist_count_per_proximity * 2} unique images, found {len(ranked_records)}."
        )

    near_pool = [dict(record, proximity="near") for record in reversed(ranked_records)]
    far_pool = [dict(record, proximity="far") for record in ranked_records]

    downloads_dir = output_root / "downloads"
    near_records, near_failures = download_shortlist(
        near_pool, args.shortlist_count_per_proximity, downloads_dir, args.base_url, args.skip_download
    )
    far_records, far_failures = download_shortlist(
        far_pool,
        args.shortlist_count_per_proximity,
        downloads_dir,
        args.base_url,
        args.skip_download,
        exclude_files={record["file_name"] for record in near_records},
    )
    annotate_brightness(near_records)
    annotate_brightness(far_records)

    near_night, near_day, near_threshold = classify_time_of_day(near_records, required_per_bin, args.seed)
    far_night, far_day, far_threshold = classify_time_of_day(far_records, required_per_bin, args.seed + 1)

    calibration_records = []
    eval_records = []
    bucket_records = {
        "night_near": near_night,
        "day_near": near_day,
        "night_far": far_night,
        "day_far": far_day,
    }

    for bucket_name, records in bucket_records.items():
        calibration_subset, eval_subset = split_for_calibration_and_eval(
            records, args.calibration_per_bin, args.eval_per_bin
        )
        for record in calibration_subset:
            calibration_records.append({**record, "bucket": bucket_name, "split": "calibration"})
        for record in eval_subset:
            eval_records.append({**record, "bucket": bucket_name, "split": "eval"})

    calibration_records = sorted(calibration_records, key=lambda record: (record["bucket"], record["file_name"]))
    eval_records = sorted(eval_records, key=lambda record: (record["bucket"], record["file_name"]))

    calibration_paths, calibration_exported = export_split("calibration", calibration_records, output_root, args.image_format)
    eval_paths, eval_exported = export_split("eval", eval_records, output_root, args.image_format)

    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    image_manifest_path = manifests_dir / "image_manifest.json"
    calibration_manifest_path = manifests_dir / "calibration_manifest.json"
    eval_manifest_path = manifests_dir / "eval_manifest.json"
    selection_records_path = manifests_dir / "selection_records.json"
    shortlist_path = manifests_dir / "shortlists.json"
    summary_path = manifests_dir / "selection_summary.json"

    all_paths = calibration_paths + eval_paths
    image_manifest_path.write_text(json.dumps(manifest_payload(all_paths), indent=2))
    calibration_manifest_path.write_text(
        json.dumps(manifest_payload(calibration_paths, source_manifest=image_manifest_path, split="calibration"), indent=2)
    )
    eval_manifest_path.write_text(
        json.dumps(manifest_payload(eval_paths, source_manifest=image_manifest_path, split="eval"), indent=2)
    )
    selection_records_path.write_text(
        json.dumps({"calibration": calibration_exported, "eval": eval_exported}, indent=2)
    )
    shortlist_path.write_text(json.dumps({"near": near_records, "far": far_records}, indent=2))

    summary = {
        "candidate_index": str(candidate_index_path),
        "output_root": str(output_root),
        "image_area_pixels": full_area,
        "shortlist_count_per_proximity": args.shortlist_count_per_proximity,
        "calibration_per_bin": args.calibration_per_bin,
        "eval_per_bin": args.eval_per_bin,
        "seed": args.seed,
        "base_url": args.base_url,
        "near_bbox_area_fraction": {
            "min": min(record["bbox_area_fraction"] for record in near_records),
            "median": statistics.median(record["bbox_area_fraction"] for record in near_records),
            "max": max(record["bbox_area_fraction"] for record in near_records),
        },
        "far_bbox_area_fraction": {
            "min": min(record["bbox_area_fraction"] for record in far_records),
            "median": statistics.median(record["bbox_area_fraction"] for record in far_records),
            "max": max(record["bbox_area_fraction"] for record in far_records),
        },
        "night_score_thresholds": {
            "near": near_threshold,
            "far": far_threshold,
        },
        "brightness_mean_summary": {
            "near": {
                "min": min(record["brightness_mean"] for record in near_records),
                "median": statistics.median(record["brightness_mean"] for record in near_records),
                "max": max(record["brightness_mean"] for record in near_records),
            },
            "far": {
                "min": min(record["brightness_mean"] for record in far_records),
                "median": statistics.median(record["brightness_mean"] for record in far_records),
                "max": max(record["brightness_mean"] for record in far_records),
            },
        },
        "night_score_summary": {
            "near": {
                "min": min(record["night_score"] for record in near_records),
                "median": statistics.median(record["night_score"] for record in near_records),
                "max": max(record["night_score"] for record in near_records),
            },
            "far": {
                "min": min(record["night_score"] for record in far_records),
                "median": statistics.median(record["night_score"] for record in far_records),
                "max": max(record["night_score"] for record in far_records),
            },
        },
        "counts": {
            "calibration": len(calibration_exported),
            "eval": len(eval_exported),
            "by_bucket": Counter(record["bucket"] for record in calibration_exported + eval_exported),
            "by_split_and_bucket": {
                "calibration": Counter(record["bucket"] for record in calibration_exported),
                "eval": Counter(record["bucket"] for record in eval_exported),
            },
        },
        "download_failures": {
            "near": near_failures,
            "far": far_failures,
        },
        "selection_mode": "bbox_area_fraction_and_brightness_proxy",
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved calibration crops to {output_root / 'crops' / 'calibration'}")
    print(f"Saved eval crops to {output_root / 'crops' / 'eval'}")
    print(f"Saved manifests to {manifests_dir}")


if __name__ == "__main__":
    main()
