import json
import math
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
    quantization_profile: str = "int8",
):
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer,
        get_symmetric_a16w8_quantization_config,
        get_symmetric_quantization_config as get_arm_symmetric_quantization_config,
    )

    compile_spec = EthosUCompatCompileSpec(
        target=target,
        system_config=system_config,
        memory_mode=memory_mode,
        config_ini=config_ini,
    )
    quantizer = EthosUQuantizer(compile_spec)
    if quantization_profile == "int8":
        quantization_config = get_arm_symmetric_quantization_config(
            is_per_channel=is_per_channel,
        )
    elif quantization_profile == "a16w8":
        quantization_config = get_symmetric_a16w8_quantization_config(
            is_per_channel=is_per_channel,
        )
    else:
        raise ValueError(f"Unsupported Ethos-U quantization profile: {quantization_profile}")
    quantizer.set_global(
        quantization_config
    )
    return quantizer, compile_spec


def export_encoder_program(encoder_only: torch.nn.Module, example_input: torch.Tensor):
    return torch.export.export(encoder_only, (example_input,))


def prepare_exported_encoder_for_ptq(
    exported_program,
    *,
    backend: str = "ethosu",
    is_per_channel: bool = True,
    quantization_profile: str = "int8",
    ethos_target: str = "ethos-u65-256",
    ethos_system_config: str | None = None,
    ethos_memory_mode: str | None = None,
    ethos_config_ini: str | None = "Arm/vela.ini",
):
    if backend == "xnnpack":
        if quantization_profile != "int8":
            raise ValueError("XNNPACK PTQ only supports the int8 quantization profile.")
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
            quantization_profile=quantization_profile,
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


def convert_encoder_after_ptq(prepared_encoder: torch.nn.Module, *, backend: str = "ethosu"):
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


def compare_latent_tensors(reference_latent: torch.Tensor, candidate_latent: torch.Tensor) -> dict:
    reference = reference_latent.detach().to("cpu", dtype=torch.float32).reshape(-1)
    candidate = candidate_latent.detach().to("cpu", dtype=torch.float32).reshape(-1)
    if reference.numel() != candidate.numel():
        raise ValueError(
            f"Latent tensors must have the same number of elements, got {reference.numel()} and {candidate.numel()}."
        )

    diff = candidate - reference
    reference_norm = torch.linalg.vector_norm(reference).item()
    candidate_norm = torch.linalg.vector_norm(candidate).item()
    l2_error = torch.linalg.vector_norm(diff).item()
    mse = torch.mean(diff * diff).item()
    rmse = math.sqrt(mse)
    cosine_similarity = torch.nn.functional.cosine_similarity(reference.unsqueeze(0), candidate.unsqueeze(0)).item()
    normalized_l2_error = l2_error / max(reference_norm, 1e-12)
    return {
        "cosine_similarity": cosine_similarity,
        "l2_error": l2_error,
        "normalized_l2_error": normalized_l2_error,
        "mse": mse,
        "rmse": rmse,
        "reference_norm": reference_norm,
        "candidate_norm": candidate_norm,
        "max_abs_error": torch.max(torch.abs(diff)).item(),
    }


def summarize_scalar_metric_records(records: list[dict], metric_names: list[str]) -> dict:
    if not records:
        return {
            metric_name: {"mean": None, "median": None, "min": None, "max": None}
            for metric_name in metric_names
        }

    summary = {}
    for metric_name in metric_names:
        values = sorted(float(record[metric_name]) for record in records)
        midpoint = len(values) // 2
        if len(values) % 2:
            median = values[midpoint]
        else:
            median = 0.5 * (values[midpoint - 1] + values[midpoint])
        summary[metric_name] = {
            "mean": sum(values) / len(values),
            "median": median,
            "min": values[0],
            "max": values[-1],
        }
    return summary


def save_token_records(
    output_path: Path,
    records: list[dict],
    *,
    repo_id: str,
    image_size: int,
    token_shape: list[int] | None = None,
    metadata: dict | None = None,
    summary: dict | None = None,
    comparisons: dict | None = None,
):
    payload = {
        "repo_id": repo_id,
        "image_size": image_size,
        "token_shape": token_shape,
        "records": records,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    if summary is not None:
        payload["summary"] = summary
    if comparisons is not None:
        payload["comparisons"] = comparisons
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
