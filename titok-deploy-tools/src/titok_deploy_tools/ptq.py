import json
from pathlib import Path

import torch
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e as convert_pt2e_torchao,
    prepare_pt2e as prepare_pt2e_torchao,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

from titok_deploy_tools.ethosu_compat import EthosUCompatCompileSpec
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


def build_ethosu_ptq_quantizer(
    *,
    target: str = "ethos-u65-256",
    system_config: str | None = None,
    memory_mode: str | None = None,
    config_ini: str | None = "Arm/vela.ini",
    is_per_channel: bool = True,
):
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer,
        get_symmetric_quantization_config as get_arm_symmetric_quantization_config,
    )

    compile_spec = EthosUCompatCompileSpec(
        target=target,
        system_config=system_config,
        memory_mode=memory_mode,
        config_ini=config_ini,
    )
    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(
        get_arm_symmetric_quantization_config(
            is_per_channel=is_per_channel,
        )
    )
    return quantizer, compile_spec


def export_encoder_program(encoder_only: torch.nn.Module, example_input: torch.Tensor):
    return torch.export.export(encoder_only, (example_input,))


def export_quantized_encoder_program(quantized_encoder: torch.nn.Module, example_input: torch.Tensor):
    return torch.export.export(quantized_encoder, (example_input,))


def prepare_exported_encoder_for_ptq(
    exported_program,
    *,
    backend: str = "xnnpack",
    is_per_channel: bool = True,
    ethos_target: str = "ethos-u65-256",
    ethos_system_config: str | None = None,
    ethos_memory_mode: str | None = None,
    ethos_config_ini: str | None = "Arm/vela.ini",
):
    if backend == "xnnpack":
        quantizer = build_xnnpack_ptq_quantizer(is_per_channel=is_per_channel)
        compile_spec = None
        graph_module = exported_program.module()
        prepare_fn = prepare_pt2e
    elif backend == "ethosu":
        quantizer, compile_spec = build_ethosu_ptq_quantizer(
            target=ethos_target,
            system_config=ethos_system_config,
            memory_mode=ethos_memory_mode,
            config_ini=ethos_config_ini,
            is_per_channel=is_per_channel,
        )
        # Arm PT2E passes do not tolerate the _guards_fn call_module inserted by
        # ExportedProgram.module() with default settings.
        graph_module = exported_program.module(check_guards=False)
        prepare_fn = prepare_pt2e_torchao
    else:
        raise ValueError(f"Unsupported PTQ backend: {backend}")
    prepared = prepare_fn(graph_module, quantizer)
    return prepared, compile_spec


def calibrate_prepared_encoder(
    prepared_encoder: torch.nn.Module,
    image_paths: list[Path],
    image_size: int,
):
    with torch.no_grad():
        for image_path in image_paths:
            image = load_image(image_path, image_size).to("cpu")
            prepared_encoder(image)


def convert_encoder_after_ptq(prepared_encoder: torch.nn.Module, *, backend: str = "xnnpack"):
    if backend == "xnnpack":
        return convert_pt2e(prepared_encoder)
    if backend == "ethosu":
        return convert_pt2e_torchao(prepared_encoder)
    raise ValueError(f"Unsupported PTQ backend: {backend}")


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
