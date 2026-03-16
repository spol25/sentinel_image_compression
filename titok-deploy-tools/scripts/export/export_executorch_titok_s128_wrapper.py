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

from titok_deploy_tools.utils import resolve_input_path, resolve_named_output, resolve_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Lower a validated TiTok S-128 .pt2 export to an ExecuTorch .pte artifact.")
    parser.add_argument(
        "--pt2-path",
        default="outputs/export/titok_s128_token_encoder.pt2",
        help="Path to the saved .pt2 export artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/export",
        help="Directory where the .pte artifact and metadata will be written.",
    )
    parser.add_argument(
        "--artifact-name",
        default="titok_s128_token_encoder.pte",
        help="Filename for the ExecuTorch .pte artifact.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from executorch.exir import to_edge
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "executorch is not installed in the active environment. "
            "Install ExecuTorch before running .pte export."
        ) from exc

    pt2_path = resolve_input_path(args.pt2_path, REPO_ROOT)
    if not pt2_path.exists():
        raise FileNotFoundError(f".pt2 artifact not found: {pt2_path}")

    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
    pte_path = resolve_named_output(output_dir, args.artifact_name)
    metadata_path = output_dir / "titok_s128_token_encoder_pte_metadata.json"

    print(f"[1/3] Loading exported program from {pt2_path}")
    exported_program = torch.export.load(pt2_path)

    print("[2/3] Lowering exported program to ExecuTorch")
    edge_program = to_edge(exported_program)
    executorch_program = edge_program.to_executorch()
    pte_path.write_bytes(executorch_program.buffer)

    print(f"[3/3] Saving ExecuTorch metadata to {metadata_path}")
    metadata = {
        "source_pt2": str(pt2_path),
        "pte_path": str(pte_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved ExecuTorch artifact to {pte_path}")


if __name__ == "__main__":
    main()
