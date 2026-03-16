import argparse
from collections import Counter
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.snapshot_serengeti import iter_bbox_candidates, write_candidate_index


def parse_args():
    parser = argparse.ArgumentParser(description="Build a Snapshot Serengeti crop candidate index from metadata and bounding boxes.")
    parser.add_argument("--metadata-json", required=True, help="Path to the main Snapshot Serengeti metadata JSON.")
    parser.add_argument("--bboxes-json", required=True, help="Path to the Snapshot Serengeti bounding box JSON.")
    parser.add_argument("--output", required=True, help="Output JSON path for the candidate index.")
    parser.add_argument("--min-bbox-size", type=int, default=32, help="Minimum width/height in pixels for retained boxes.")
    parser.add_argument(
        "--margin-fraction",
        type=float,
        default=0.05,
        help="Fractional margin to add around each bounding box before cropping.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_path = Path(args.metadata_json).expanduser().resolve()
    bboxes_path = Path(args.bboxes_json).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    candidates = list(
        iter_bbox_candidates(
            metadata_path,
            bboxes_path,
            min_bbox_size=args.min_bbox_size,
            margin_fraction=args.margin_fraction,
        )
    )
    if not candidates:
        raise SystemExit("No bounding-box crop candidates found.")

    category_counts = Counter(candidate.category_name for candidate in candidates if candidate.category_name)
    location_count = len({candidate.location for candidate in candidates if candidate.location is not None})
    season_count = len({candidate.season for candidate in candidates if candidate.season is not None})
    metadata = {
        "metadata_json": str(metadata_path),
        "bboxes_json": str(bboxes_path),
        "min_bbox_size": args.min_bbox_size,
        "margin_fraction": args.margin_fraction,
        "unique_locations": location_count,
        "unique_seasons": season_count,
        "top_categories": category_counts.most_common(25),
    }
    write_candidate_index(output_path, candidates, metadata)
    print(f"Saved Snapshot Serengeti candidate index with {len(candidates)} boxes to {output_path}")


if __name__ == "__main__":
    main()
