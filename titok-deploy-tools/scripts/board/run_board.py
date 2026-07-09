#!/usr/bin/env python3
"""Single entrypoint for board utility and experiment scripts."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


LEGACY_COMMANDS = {
    "audit-activation-saturation": "audit_activation_saturation.py",
    "compare-board-capture": "compare_board_capture.py",
    "controlled-block0-deep-dive": "controlled_block0_deep_dive_host.py",
    "controlled-host-capture-and-lower": "controlled_host_capture_and_lower.py",
    "controlled-mobilenet-host-capture-and-lower": "controlled_mobilenet_host_capture_and_lower.py",
    "controlled-prologue-block0-boundary": "controlled_prologue_block0_boundary_host.py",
    "full-bhld-vq-eval": "full_bhld_vq_eval_host.py",
    "gelu-range-widening": "run_gelu_range_widening_experiments.py",
    "run-cm33-batch": "run_cm33_batch_from_host_summary.py",
}


def _legacy_script_path(script_name: str) -> Path:
    legacy_path = SCRIPT_DIR / "legacy" / script_name
    if legacy_path.exists():
        return legacy_path
    return SCRIPT_DIR / script_name


def _run_legacy(command: str, argv: list[str]) -> None:
    script_path = _legacy_script_path(LEGACY_COMMANDS[command])
    if not script_path.exists():
        raise SystemExit(f"Legacy board command implementation not found: {script_path}")
    sys.argv = [str(script_path), *argv]
    runpy.run_path(str(script_path), run_name="__main__")


def _load_image_tensor(path: Path, image_size: int):
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(image_np, (2, 0, 1))[None, ...])


def _cmd_make_cm33_input(argv: list[str]) -> None:
    from titok_deploy_tools.board_tools.cm33 import write_input_blob

    parser = argparse.ArgumentParser(description="Create a CM33 DDR input blob from an image.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args(argv)

    tensor = _load_image_tensor(args.image, args.image_size)
    metadata = write_input_blob(args.output, tensor, metadata_path=args.metadata, source=str(args.image))
    print(json.dumps(metadata, indent=2))


def _cmd_parse_cm33_output(argv: list[str]) -> None:
    import numpy as np

    from titok_deploy_tools.board_tools.cm33 import parse_output_blob

    parser = argparse.ArgumentParser(description="Parse a CM33 output blob into npz/json artifacts.")
    parser.add_argument("--blob", required=True, type=Path)
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args(argv)

    output, metadata = parse_output_blob(args.blob)
    args.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.npz, output=output)
    metadata["npz"] = str(args.npz)
    metadata_path = args.metadata or args.npz.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = ["make-cm33-input", "parse-cm33-output", *sorted(LEGACY_COMMANDS)]
    parser.add_argument("command", choices=subcommands)
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for the selected command.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "make-cm33-input":
        _cmd_make_cm33_input(args.args)
    elif args.command == "parse-cm33-output":
        _cmd_parse_cm33_output(args.args)
    else:
        _run_legacy(args.command, args.args)


if __name__ == "__main__":
    main()
