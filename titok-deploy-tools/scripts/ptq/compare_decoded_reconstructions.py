import argparse
import json
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titok_deploy_tools.decode import decode_token_ids
from titok_deploy_tools.titok_env import add_titok_root_to_path
from titok_deploy_tools.utils import resolve_input_path, resolve_named_output, resolve_output_dir, save_reconstruction


def parse_args():
    parser = argparse.ArgumentParser(description="Decode and compare two token output files.")
    parser.add_argument("--titok-root", required=True, help="Path to a separate 1d-tokenizer checkout.")
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_s128_imagenet",
        help="Hugging Face repo for the pretrained TiTok-S-128 tokenizer.",
    )
    parser.add_argument("--reference", required=True, help="Reference token JSON path.")
    parser.add_argument("--candidate", required=True, help="Candidate token JSON path.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory where decoded images or a standalone summary will be written.",
    )
    parser.add_argument(
        "--summary-name",
        default=None,
        help="Optional filename for a standalone reconstruction comparison JSON.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Write decoded reference/candidate PNGs alongside the summary.",
    )
    return parser.parse_args()


def load_payload(path: Path):
    return json.loads(path.read_text())


def load_records(path: Path):
    payload = load_payload(path)
    if "records" not in payload:
        raise ValueError(f"Expected record-based token payload in {path}")
    return payload, payload["records"]


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def main():
    args = parse_args()
    add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    reference_path = resolve_input_path(args.reference, REPO_ROOT)
    candidate_path = resolve_input_path(args.candidate, REPO_ROOT)
    _, reference_records = load_records(reference_path)
    candidate_payload, candidate_records = load_records(candidate_path)
    if len(reference_records) != len(candidate_records):
        raise ValueError("Reference and candidate token files have different record counts.")

    tokenizer = TiTok.from_pretrained(args.repo_id).eval().to("cpu")
    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir) if args.output_dir else None

    results = []
    for index, (ref_record, cand_record) in enumerate(zip(reference_records, candidate_records)):
        ref_tokens = torch.tensor(ref_record["tokens"], dtype=torch.long)
        cand_tokens = torch.tensor(cand_record["tokens"], dtype=torch.long)
        ref_image = decode_token_ids(tokenizer, ref_tokens, "cpu")
        cand_image = decode_token_ids(tokenizer, cand_tokens, "cpu")
        mse = torch.mean((ref_image - cand_image) ** 2).item()
        psnr = psnr_from_mse(mse)

        result = {
            "image": ref_record["image"],
            "mse": mse,
            "psnr": psnr,
        }
        if args.save_images:
            if output_dir is None:
                raise ValueError("--save-images requires --output-dir")
            ref_out = resolve_named_output(output_dir, f"reference_decode_{index:03d}.png")
            cand_out = resolve_named_output(output_dir, f"candidate_decode_{index:03d}.png")
            save_reconstruction(ref_image, ref_out)
            save_reconstruction(cand_image, cand_out)
            result["reference_decode"] = str(ref_out)
            result["candidate_decode"] = str(cand_out)
        results.append(result)

    summary = {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "num_images": len(results),
        "mean_mse": sum(item["mse"] for item in results) / len(results),
        "mean_psnr": sum(item["psnr"] for item in results) / len(results),
        "results": results,
    }
    comparisons = candidate_payload.get("comparisons", {})
    if not isinstance(comparisons, dict):
        comparisons = {}
    comparisons["reconstructions"] = summary
    candidate_payload["comparisons"] = comparisons
    candidate_path.write_text(json.dumps(candidate_payload, indent=2))

    if args.output_dir and args.summary_name:
        summary_path = resolve_named_output(output_dir, args.summary_name)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved reconstruction comparison summary to {summary_path}")
    else:
        print(
            "Embedded reconstruction comparison in "
            f"{candidate_path} with mean PSNR {summary['mean_psnr']:.6f}"
        )


if __name__ == "__main__":
    main()
