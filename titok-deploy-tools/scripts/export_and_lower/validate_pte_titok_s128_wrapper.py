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

from titok_deploy_tools.utils import load_image, resolve_input_path, resolve_named_output, resolve_output_dir
from titok_deploy_tools.wrappers import TiTokTokenEncoder
from titok_deploy_tools.titok_env import add_titok_root_to_path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate an ExecuTorch .pte artifact against eager and .pt2 references.")
    parser.add_argument("--titok-root", required=True, help="Path to a separate 1d-tokenizer checkout.")
    parser.add_argument(
        "--repo-id",
        default="yucornetto/tokenizer_titok_s128_imagenet",
        help="Hugging Face repo for the pretrained TiTok-S-128 tokenizer.",
    )
    parser.add_argument(
        "--pt2-path",
        default="outputs/export/titok_s128_token_encoder.pt2",
        help="Path to the saved .pt2 export artifact.",
    )
    parser.add_argument(
        "--pte-path",
        default="outputs/export/titok_s128_token_encoder.pte",
        help="Path to the saved .pte artifact.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Input image path used for validation. Defaults to a sample image in --titok-root/assets.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/test_s128",
        help="Directory where validation token JSON will be written.",
    )
    parser.add_argument(
        "--tokens-output",
        default="titok_s128_pte_tokens.json",
        help="Filename for serialized .pte token output inside --output-dir.",
    )
    return parser.parse_args()


def save_tokens(tokens: torch.Tensor, path: Path, repo_id: str):
    payload = {
        "repo_id": repo_id,
        "shape": list(tokens.shape),
        "tokens": tokens.detach().to("cpu").tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def normalize_runtime_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        if len(output) != 1:
            raise ValueError(f"Expected a single tensor output, got {len(output)} outputs")
        first = output[0]
        if not isinstance(first, torch.Tensor):
            raise TypeError(f"Expected tensor output, got {type(first)}")
        return first
    raise TypeError(f"Unsupported runtime output type: {type(output)}")


def main():
    args = parse_args()
    titok_root = add_titok_root_to_path(args.titok_root)
    from modeling.titok import TiTok

    try:
        from executorch.runtime import Runtime
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "executorch is not installed in the active environment. "
            "Install ExecuTorch before running .pte validation."
        ) from exc

    pt2_path = resolve_input_path(args.pt2_path, REPO_ROOT)
    pte_path = resolve_input_path(args.pte_path, REPO_ROOT)
    if not pt2_path.exists():
        raise FileNotFoundError(f".pt2 artifact not found: {pt2_path}")
    if not pte_path.exists():
        raise FileNotFoundError(f".pte artifact not found: {pte_path}")

    if args.image is None:
        image_path = titok_root / "assets" / "ILSVRC2012_val_00010240.png"
    else:
        image_path = resolve_input_path(args.image)

    output_dir = resolve_output_dir(REPO_ROOT, args.output_dir)
    tokens_output_path = resolve_named_output(output_dir, args.tokens_output)

    print(f"[1/5] Loading TiTok model from {args.repo_id} on CPU for eager reference")
    titok = TiTok.from_pretrained(args.repo_id)
    titok.eval()
    titok.requires_grad_(False)
    titok = titok.to("cpu")

    print("[2/5] Building eager wrapper and example input")
    wrapper = TiTokTokenEncoder(titok)
    wrapper.eval()
    wrapper.requires_grad_(False)
    wrapper = wrapper.to("cpu")
    image_size = int(titok.config.dataset.preprocessing.crop_size)
    example_input = load_image(image_path, image_size).to("cpu")
    print(
        f"example_input shape={tuple(example_input.shape)} "
        f"stride={example_input.stride()} "
        f"contiguous={example_input.is_contiguous()} "
        f"channels_last={example_input.is_contiguous(memory_format=torch.channels_last)}"
    )

    eager_tokens = wrapper(example_input)
    pt2_tokens = torch.export.load(pt2_path).module()(example_input)

    print(f"[3/5] Loading ExecuTorch runtime and program from {pte_path}")
    runtime = Runtime.get()
    program = runtime.load_program(str(pte_path))
    method = program.load_method("forward")

    print("[4/5] Executing .pte and validating parity against eager and .pt2")
    pte_output = method.execute((example_input,))
    print(f"runtime output type={type(pte_output)}")
    pte_tokens = normalize_runtime_output(pte_output)
    if not torch.equal(eager_tokens, pt2_tokens):
        raise SystemExit("pt2 reference output does not match eager wrapper output.")
    if not torch.equal(eager_tokens, pte_tokens):
        raise SystemExit(".pte runtime output does not match eager wrapper output.")

    print(f"[5/5] Saving .pte token output to {tokens_output_path}")
    save_tokens(pte_tokens, tokens_output_path, args.repo_id)
    print("pte validation passed.")


if __name__ == "__main__":
    main()
