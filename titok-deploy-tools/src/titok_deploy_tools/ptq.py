import json
from pathlib import Path

import torch
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

from titok_deploy_tools.utils import load_image
from titok_deploy_tools.wrappers import TiTokEncoderOnly, TiTokTokenEncoder, TiTokVectorQuantizerTokens


def load_manifest_records(manifest_path: Path) -> list[Path]:
    manifest = json.loads(manifest_path.read_text())
    return [Path(image_path) for image_path in manifest["images"]]


def build_encoder_quantizer_split(titok, flatten_output: bool = True):
    encoder_only = TiTokEncoderOnly(titok)
    latents_to_tokens = TiTokVectorQuantizerTokens(titok, flatten_output=flatten_output)
    full_wrapper = TiTokTokenEncoder(titok, flatten_output=flatten_output)
    return encoder_only, latents_to_tokens, full_wrapper


def build_xnnpack_ptq_quantizer(is_per_channel: bool = True, is_qat: bool = False):
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(
        get_symmetric_quantization_config(
            is_per_channel=is_per_channel,
            is_qat=is_qat,
        )
    )
    return quantizer


def export_encoder_program(encoder_only: torch.nn.Module, example_input: torch.Tensor):
    return torch.export.export(encoder_only, (example_input,))


def prepare_exported_encoder_for_ptq(
    exported_program,
    *,
    is_per_channel: bool = True,
):
    quantizer = build_xnnpack_ptq_quantizer(is_per_channel=is_per_channel)
    prepared = prepare_pt2e(exported_program.module(), quantizer)
    return prepared


def calibrate_prepared_encoder(
    prepared_encoder: torch.nn.Module,
    image_paths: list[Path],
    image_size: int,
):
    with torch.no_grad():
        for image_path in image_paths:
            image = load_image(image_path, image_size).to("cpu")
            prepared_encoder(image)


def convert_encoder_after_ptq(prepared_encoder: torch.nn.Module):
    return convert_pt2e(prepared_encoder)


def run_encoder_with_float_quantizer(
    encoder_module: torch.nn.Module,
    latents_to_tokens: torch.nn.Module,
    image: torch.Tensor,
) -> torch.Tensor:
    latent = encoder_module(image)
    return latents_to_tokens(latent)


def save_token_records(
    output_path: Path,
    records: list[dict],
    *,
    repo_id: str,
    image_size: int,
    token_shape: list[int] | None = None,
    metadata: dict | None = None,
):
    payload = {
        "repo_id": repo_id,
        "image_size": image_size,
        "token_shape": token_shape,
        "records": records,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def summarize_token_records(records: list[dict]) -> dict:
    unique_token_ids = set()
    token_count = None
    for record in records:
        token_list = record["tokens"]
        unique_token_ids.update(token_list)
        token_count = len(token_list)

    if token_count is None:
        return {
            "num_images": 0,
            "token_count_per_image": 0,
            "num_unique_token_ids": 0,
            "min_token_id": None,
            "max_token_id": None,
        }

    return {
        "num_images": len(records),
        "token_count_per_image": token_count,
        "num_unique_token_ids": len(unique_token_ids),
        "min_token_id": min(unique_token_ids),
        "max_token_id": max(unique_token_ids),
    }
