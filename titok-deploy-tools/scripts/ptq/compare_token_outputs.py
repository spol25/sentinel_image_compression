import argparse
import json
import math
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.utils import resolve_input_path, resolve_named_output, resolve_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Compare float vs candidate token outputs.")
    parser.add_argument("--reference", required=True, help="Reference token JSON path.")
    parser.add_argument("--candidate", required=True, help="Candidate token JSON path.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory where a standalone comparison JSON will be written.",
    )
    parser.add_argument(
        "--summary-name",
        default=None,
        help="Optional filename for a standalone comparison JSON.",
    )
    return parser.parse_args()


def load_payload(path: Path):
    return json.loads(path.read_text())


def load_records(path: Path):
    payload = load_payload(path)
    return payload, extract_records(payload)


def extract_records(payload: dict):
    if "records" in payload:
        return payload["records"]
    raise ValueError(f"Unrecognized token payload format in {path}")


def median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return 0.5 * (values[midpoint - 1] + values[midpoint])


def main():
    args = parse_args()
    reference_path = resolve_input_path(args.reference, REPO_ROOT)
    candidate_path = resolve_input_path(args.candidate, REPO_ROOT)

    reference_payload, reference_records = load_records(reference_path)
    candidate_payload, candidate_records = load_records(candidate_path)
    if len(reference_records) != len(candidate_records):
        raise ValueError("Reference and candidate token files have different record counts.")

    changed_images = []
    per_position_matches = None
    total_tokens = 0
    total_matches = 0
    reference_histogram = {}
    candidate_histogram = {}
    for ref_record, cand_record in zip(reference_records, candidate_records):
        ref_tokens = ref_record["tokens"]
        cand_tokens = cand_record["tokens"]
        if len(ref_tokens) != len(cand_tokens):
            raise ValueError(f"Token length mismatch for {ref_record['image']}")
        if per_position_matches is None:
            per_position_matches = [0] * len(ref_tokens)
        matches = sum(int(a == b) for a, b in zip(ref_tokens, cand_tokens))
        for index, (a, b) in enumerate(zip(ref_tokens, cand_tokens)):
            per_position_matches[index] += int(a == b)
            reference_histogram[a] = reference_histogram.get(a, 0) + 1
            candidate_histogram[b] = candidate_histogram.get(b, 0) + 1
        total_tokens += len(ref_tokens)
        total_matches += matches
        changed_count = len(ref_tokens) - matches
        if changed_count:
            changed_images.append(
                {
                    "image": ref_record["image"],
                    "changed_tokens": changed_count,
                    "token_agreement": matches / len(ref_tokens),
                }
            )

    changed_counts = [item["changed_tokens"] for item in changed_images]
    num_images = len(reference_records)
    token_count_per_image = len(reference_records[0]["tokens"]) if reference_records else 0
    per_position_token_agreement = (
        [matches / num_images for matches in per_position_matches] if per_position_matches is not None and num_images else []
    )
    union_token_ids = sorted(set(reference_histogram) | set(candidate_histogram))
    reference_freq = [reference_histogram.get(token_id, 0) / total_tokens for token_id in union_token_ids]
    candidate_freq = [candidate_histogram.get(token_id, 0) / total_tokens for token_id in union_token_ids]
    histogram_l1 = sum(abs(a - b) for a, b in zip(reference_freq, candidate_freq))
    top_histogram_drift = sorted(
        (
            {
                "token_id": token_id,
                "reference_frequency": reference_histogram.get(token_id, 0) / total_tokens,
                "candidate_frequency": candidate_histogram.get(token_id, 0) / total_tokens,
                "absolute_frequency_diff": abs(
                    reference_histogram.get(token_id, 0) / total_tokens
                    - candidate_histogram.get(token_id, 0) / total_tokens
                ),
            }
            for token_id in union_token_ids
        ),
        key=lambda item: item["absolute_frequency_diff"],
        reverse=True,
    )[:25]

    summary = {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "num_images": num_images,
        "token_count_per_image": token_count_per_image,
        "total_tokens": total_tokens,
        "total_matches": total_matches,
        "overall_token_agreement": total_matches / total_tokens if total_tokens else 0.0,
        "num_changed_images": len(changed_images),
        "changed_token_count_stats": {
            "mean": sum(changed_counts) / len(changed_counts) if changed_counts else 0.0,
            "median": median(changed_counts),
            "min": min(changed_counts) if changed_counts else 0,
            "max": max(changed_counts) if changed_counts else 0,
        },
        "per_position_token_agreement": per_position_token_agreement,
        "token_histogram_drift": {
            "l1_distance": histogram_l1,
            "total_variation_distance": 0.5 * histogram_l1,
            "sqrt_js_proxy": math.sqrt(max(0.0, 0.5 * histogram_l1)),
            "top_token_frequency_drift": top_histogram_drift,
        },
        "changed_images": changed_images,
    }

    comparisons = candidate_payload.get("comparisons", {})
    if not isinstance(comparisons, dict):
        comparisons = {}
    comparisons["tokens"] = summary
    candidate_payload["comparisons"] = comparisons
    candidate_path.write_text(json.dumps(candidate_payload, indent=2))

    if args.output_dir and args.summary_name:
        output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
        summary_path = resolve_named_output(output_dir, args.summary_name)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved token comparison summary to {summary_path}")
    else:
        print(
            "Embedded token comparison in "
            f"{candidate_path} with overall agreement {summary['overall_token_agreement']:.6f}"
        )


if __name__ == "__main__":
    main()
