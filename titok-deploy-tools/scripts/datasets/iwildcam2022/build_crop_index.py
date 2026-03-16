import argparse
from collections import Counter
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.iwildcam2022 import iter_crop_candidates, write_crop_index


def parse_args():
    parser = argparse.ArgumentParser(description="Index iWildCam 2022 object crop candidates from MegaDetector + instance masks.")
    parser.add_argument("--dataset-root", required=True, help="Path to the extracted iWildCam 2022 dataset root.")
    parser.add_argument(
        "--output",
        default="crop_index.json",
        help="Output JSON path for the crop candidate index.",
    )
    parser.add_argument("--min-conf", type=float, default=0.2, help="Minimum MegaDetector confidence.")
    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=32,
        help="Minimum width/height in pixels for retained crops.",
    )
    parser.add_argument(
        "--margin-fraction",
        type=float,
        default=0.05,
        help="Fractional margin to add around the detected object crop.",
    )
    parser.add_argument(
        "--include-non-animals",
        action="store_true",
        help="Keep non-animal detections as well. By default only MegaDetector animal category 1 is kept.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    candidates = list(
        iter_crop_candidates(
            dataset_root,
            min_conf=args.min_conf,
            animal_only=not args.include_non_animals,
            min_crop_size=args.min_crop_size,
            margin_fraction=args.margin_fraction,
        )
    )
    if not candidates:
        raise SystemExit("No crop candidates found with the requested filters.")

    split_counts = Counter(candidate.source_split for candidate in candidates)
    labeled_count = sum(candidate.category_id is not None for candidate in candidates)
    unique_locations = len({candidate.location for candidate in candidates if candidate.location is not None})
    unique_categories = len({candidate.category_id for candidate in candidates if candidate.category_id is not None})

    metadata = {
        "dataset_root": str(dataset_root),
        "min_conf": args.min_conf,
        "animal_only": not args.include_non_animals,
        "min_crop_size": args.min_crop_size,
        "margin_fraction": args.margin_fraction,
        "split_counts": dict(split_counts),
        "labeled_count": labeled_count,
        "unique_locations": unique_locations,
        "unique_categories": unique_categories,
    }
    write_crop_index(output_path, candidates, metadata)
    print(f"Saved crop index with {len(candidates)} candidates to {output_path}")


if __name__ == "__main__":
    main()
