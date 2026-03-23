import argparse
import json
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
    if "tokens" in payload:
        return [{"image": payload.get("image", str(path)), "tokens": payload["tokens"][0]}]
    raise ValueError(f"Unrecognized token payload format in {path}")


def main():
    args = parse_args()
    reference_path = resolve_input_path(args.reference, REPO_ROOT)
    candidate_path = resolve_input_path(args.candidate, REPO_ROOT)

    reference_payload, reference_records = load_records(reference_path)
    candidate_payload, candidate_records = load_records(candidate_path)
    if len(reference_records) != len(candidate_records):
        raise ValueError("Reference and candidate token files have different record counts.")

    changed_images = []
    total_tokens = 0
    total_matches = 0
    for ref_record, cand_record in zip(reference_records, candidate_records):
        ref_tokens = ref_record["tokens"]
        cand_tokens = cand_record["tokens"]
        if len(ref_tokens) != len(cand_tokens):
            raise ValueError(f"Token length mismatch for {ref_record['image']}")
        matches = sum(int(a == b) for a, b in zip(ref_tokens, cand_tokens))
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

    summary = {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "num_images": len(reference_records),
        "total_tokens": total_tokens,
        "total_matches": total_matches,
        "overall_token_agreement": total_matches / total_tokens if total_tokens else 0.0,
        "num_changed_images": len(changed_images),
        "changed_images": changed_images,
    }

    comparisons = candidate_payload.get("comparisons", [])
    comparisons = [
        item
        for item in comparisons
        if not (
            item.get("reference") == str(reference_path)
            and item.get("candidate") == str(candidate_path)
        )
    ]
    comparisons.append(summary)
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
