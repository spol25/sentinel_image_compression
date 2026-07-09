import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "titok_vs_fallback_step_by_step_comparison"
DEFAULT_INPUT = REPO_ROOT / "outputs" / "better_inputs" / "0026_day_near_S2_H08_R3_IMAG0666.jpg"
os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / "outputs" / ".torch_cache"))


CANDIDATES = {
    "source_matmul_attention": {
        "label": "Source matmul attention",
        "image": "source_matmul_token_encoder_reconstruction.png",
    },
    "fallback_original_non_quantized": {
        "label": "Fallback original non-quantized",
        "image": "fallback_original_non_quantized_reconstruction.png",
    },
    "quantized_full_a16w8": {
        "label": "Quantized full A16W8",
        "image": "quantized_full_a16w8_reconstruction.png",
    },
    "quantized_residual_add_a16w8": {
        "label": "Quantized residual-add A16W8",
        "image": "quantized_residual_add_a16w8_reconstruction.png",
    },
    "quantized_fallback_qnnpack": {
        "label": "Quantized fallback QNNPACK",
        "image": "quantized_fallback_qnnpack_reconstruction.png",
    },
    "quantized_fallback_tflite_pre_lowered": {
        "label": "Quantized fallback TFLite pre-lowered",
        "image": "quantized_fallback_tflite_pre_lowered_reconstruction.png",
    },
    "quantized_fallback_ai_edge_pre_lowered": {
        "label": "Quantized fallback ai_edge_torch pre-lowered",
        "image": "quantized_fallback_ai_edge_pre_lowered_reconstruction.png",
    },
    "quantized_fallback_tflite_a16w8_pre_lowered": {
        "label": "Quantized fallback TFLite A16W8 pre-lowered",
        "image": "quantized_fallback_tflite_a16w8_pre_lowered_reconstruction.png",
    },
    "quantized_fallback_apr9_pre_lowered": {
        "label": "Quantized fallback Apr 9 pre-lowered TFLite",
        "image": "apr9_pre_lowered_reconstruction.jpg",
    },
    "quantized_fallback_vela_lowered_board": {
        "label": "Quantized fallback Vela lowered board",
        "image": "vela_lowered_board_reconstruction.jpg",
    },
}


TOKEN_FILES = {
    "default_titok_token_encoder": "default_titok_token_encoder_tokens.json",
    "source_matmul_attention": "source_matmul_token_encoder_tokens.json",
    "fallback_original_non_quantized": "fallback_original_non_quantized_tokens.json",
    "quantized_full_a16w8": "quantized_full_a16w8_tokens.json",
    "quantized_residual_add_a16w8": "quantized_residual_add_a16w8_tokens.json",
    "quantized_fallback_qnnpack": "quantized_fallback_qnnpack_tokens.json",
    "quantized_fallback_tflite_pre_lowered": "quantized_fallback_tflite_pre_lowered_tokens.json",
    "quantized_fallback_ai_edge_pre_lowered": "quantized_fallback_ai_edge_pre_lowered_tokens.json",
    "quantized_fallback_tflite_a16w8_pre_lowered": "quantized_fallback_tflite_a16w8_pre_lowered_tokens.json",
    "quantized_fallback_apr9_pre_lowered": "apr9_pre_lowered_tokens.json",
    "quantized_fallback_vela_lowered_board": "vela_lowered_board_tokens.json",
}

REPORT_OUTPUT_FILES = [
    "source_matmul_token_encoder_tokens.json",
    "source_matmul_token_encoder_reconstruction.png",
    "fallback_original_non_quantized_tokens.json",
    "fallback_original_non_quantized_reconstruction.png",
    "quantized_full_a16w8_tokens.json",
    "quantized_full_a16w8_reconstruction.png",
    "quantized_residual_add_a16w8_tokens.json",
    "quantized_residual_add_a16w8_reconstruction.png",
    "quantized_fallback_qnnpack_tokens.json",
    "quantized_fallback_qnnpack_reconstruction.png",
    "quantized_fallback_tflite_pre_lowered_tokens.json",
    "quantized_fallback_tflite_pre_lowered_reconstruction.png",
    "quantized_fallback_tflite_pre_lowered_run.json",
    "quantized_fallback_ai_edge_pre_lowered_tokens.json",
    "quantized_fallback_ai_edge_pre_lowered_reconstruction.png",
    "quantized_fallback_ai_edge_pre_lowered_run.json",
    "quantized_fallback_tflite_a16w8_pre_lowered_tokens.json",
    "quantized_fallback_tflite_a16w8_pre_lowered_reconstruction.png",
    "quantized_fallback_tflite_a16w8_pre_lowered_run.json",
    "apr9_pre_lowered_tokens.json",
    "apr9_pre_lowered_tokens.bin",
    "apr9_pre_lowered_reconstruction.jpg",
    "apr9_pre_lowered_run.json",
    "vela_lowered_board_tokens.json",
    "vela_lowered_board_tokens.bin",
    "vela_lowered_board_reconstruction.jpg",
    "vela_lowered_board_run.json",
    "quantized_pre_lowered_run.json",
    "input_source_matmul_fallback_stitched.png",
    "metrics.json",
    "step_by_step_metrics.md",
]

HIDDEN_WRAPPERS = {
    "default_titok_token_encoder",
    "reconstruct_titok_example_s128",
}

_LPIPS_ALEX = None


def load_rgb(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def windowed_ssim_components(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels = reference.shape[1]
    window = gaussian_window(window_size, sigma, channels, reference.device)
    padding = window_size // 2
    c1 = 0.01**2
    c2 = 0.03**2

    x = F.pad(reference, (padding, padding, padding, padding), mode="reflect")
    y = F.pad(candidate, (padding, padding, padding, padding), mode="reflect")

    mux = F.conv2d(x, window, groups=channels)
    muy = F.conv2d(y, window, groups=channels)
    mux2 = mux * mux
    muy2 = muy * muy
    muxy = mux * muy

    sigmax2 = F.conv2d(x * x, window, groups=channels) - mux2
    sigmay2 = F.conv2d(y * y, window, groups=channels) - muy2
    sigmaxy = F.conv2d(x * y, window, groups=channels) - muxy

    luminance = (2.0 * muxy + c1) / (mux2 + muy2 + c1)
    contrast_structure = (2.0 * sigmaxy + c2) / (sigmax2 + sigmay2 + c2)
    ssim_map = luminance * contrast_structure
    return ssim_map.mean(), contrast_structure.mean()


def windowed_ssim(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float(windowed_ssim_components(reference, candidate)[0].clamp(0.0, 1.0).item())


def ms_ssim(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float32)
    x = reference
    y = candidate
    mcs = []
    for _ in range(len(weights) - 1):
        ssim_value, cs_value = windowed_ssim_components(x, y)
        mcs.append(cs_value.clamp(min=1e-6, max=1.0))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        y = F.avg_pool2d(y, kernel_size=2, stride=2)
    ssim_value, _ = windowed_ssim_components(x, y)
    values = torch.stack(mcs + [ssim_value.clamp(min=1e-6, max=1.0)])
    return float(torch.prod(values ** weights).item())


def maybe_lpips(reference: torch.Tensor, candidate: torch.Tensor) -> float | None:
    global _LPIPS_ALEX
    try:
        import lpips  # type: ignore
    except ImportError:
        return None

    if _LPIPS_ALEX is None:
        _LPIPS_ALEX = lpips.LPIPS(net="alex")
        _LPIPS_ALEX.eval()
    with torch.no_grad():
        reference_lpips = reference * 2.0 - 1.0
        candidate_lpips = candidate * 2.0 - 1.0
        return float(_LPIPS_ALEX(reference_lpips, candidate_lpips).item())


def image_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    diff = candidate - reference
    abs_diff = diff.abs()
    sq_diff = diff * diff
    mse = float(sq_diff.mean().item())
    channel_names = ("r", "g", "b")
    per_channel = {}
    for idx, name in enumerate(channel_names):
        channel_diff = diff[:, idx]
        channel_abs = channel_diff.abs()
        channel_mse = float((channel_diff * channel_diff).mean().item())
        per_channel[name] = {
            "mse": channel_mse,
            "rmse": math.sqrt(channel_mse),
            "mae": float(channel_abs.mean().item()),
            "bias": float(channel_diff.mean().item()),
        }

    metrics = {
        "psnr": psnr_from_mse(mse),
        "ssim_windowed": windowed_ssim(reference, candidate),
        "ms_ssim": ms_ssim(reference, candidate),
        "lpips_alex": maybe_lpips(reference, candidate),
        "mse": mse,
        "mae": float(abs_diff.mean().item()),
        "rmse": math.sqrt(mse),
        "p95_abs_error": float(torch.quantile(abs_diff.reshape(-1), 0.95).item()),
        "p99_abs_error": float(torch.quantile(abs_diff.reshape(-1), 0.99).item()),
        "max_abs_error": float(abs_diff.max().item()),
        "per_channel_error": per_channel,
    }
    return metrics


def load_tokens(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return torch.tensor(payload["tokens"], dtype=torch.long)


def token_agreement(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.reshape(-1)
    cand = candidate.reshape(-1)
    if ref.numel() != cand.numel():
        raise ValueError(f"Token shapes differ: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
    matches = ref == cand
    return {
        "shape_reference": list(reference.shape),
        "shape_candidate": list(candidate.shape),
        "exact_match": bool(matches.all().item()),
        "exact_fraction": float(matches.to(torch.float32).mean().item()),
        "mismatch_count": int((~matches).sum().item()),
        "num_tokens": int(ref.numel()),
        "unique_reference_tokens": int(torch.unique(ref).numel()),
        "unique_candidate_tokens": int(torch.unique(cand).numel()),
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.{digits}g}"
    return str(value)


def display_path(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def markdown(metrics: dict[str, Any], output_dir: Path) -> str:
    wrappers = metrics["wrappers"]
    lines = [
        "# TiTok vs Source-Matmul vs Fallback Step-by-Step Comparison",
        "",
        f"- Input image: `{display_path(metrics['image'])}`",
        f"- Image size: `{metrics['image_size']}x{metrics['image_size']}`",
    ]
    fallback = wrappers.get("fallback_original_non_quantized", {})
    if "checkpoint" in fallback:
        lines.append(f"- Fallback checkpoint: `{display_path(fallback['checkpoint'])}`")
    lines.extend(
        [
            "",
            "## Wrapper Used Previously",
            "",
            "The first run used `scripts/validate_titok_s128_wrapper.py`, which instantiates "
            "`TiTokTokenEncoder(titok)`. That is the default/full TiTok token wrapper, not the "
            "source-matmul variant.",
            "",
            "## Output Files",
            "",
        ]
    )
    for filename in REPORT_OUTPUT_FILES:
        if (output_dir / filename).exists():
            lines.append(f"- `{filename}`")

    lines.extend(
        [
            "",
            "## Input Image vs Reconstruction",
            "",
            "| Model path | PSNR | MS-SSIM | LPIPS alex | MAE | p95 abs err | p99 abs err |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, spec in CANDIDATES.items():
        if key not in wrappers:
            continue
        image_stats = wrappers[key]["input_vs_reconstruction"]
        lines.append(
            "| "
            + " | ".join(
                [
                    spec["label"],
                    fmt(image_stats["psnr"]),
                    fmt(image_stats["ms_ssim"]),
                    fmt(image_stats["lpips_alex"]),
                    fmt(image_stats["mae"]),
                    fmt(image_stats["p95_abs_error"]),
                    fmt(image_stats["p99_abs_error"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Original TiTok Encoder Reference vs Encoder Latents",
            "",
            "| Wrapper encoder | Cosine similarity | Normalized L2 error | Max abs error |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    latent_rows = (
        ("source_matmul_attention", "Source matmul encoder wrapper"),
        ("quantized_full_a16w8", "Quantized full A16W8 encoder"),
        ("quantized_residual_add_a16w8", "Quantized residual-add A16W8 encoder"),
    )
    for key, label in latent_rows:
        if key not in wrappers:
            continue
        latent = wrappers[key].get("latent_vs_titok_encode_reference", {})
        if not latent:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    fmt(latent.get("cosine_similarity")),
                    fmt(latent.get("normalized_l2_error")),
                    fmt(latent.get("max_abs_error")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "The quantized TiTok encoder rows compare against the original TiTok encoder latent before VQ, "
        "which is the encoder output used inside `TiTok.encode()`. The fallback model does not expose that "
        "same intermediate encoder latent in its existing script interface, so it is compared by "
        "reconstruction metrics and token agreement instead."
    )

    lines.extend(
        [
            "",
            "## VQ Token Agreement",
            "",
            "| Comparison | Exact fraction | Mismatches |",
            "| --- | ---: | ---: |",
        ]
    )
    token_rows = [
        ("TiTok.encode vs source matmul wrapper", wrappers["source_matmul_attention"].get("tokens_vs_titok_encode_reference")),
        (
            "Source matmul vs fallback original non-quantized",
            metrics.get("source_matmul_vs_fallback_original_non_quantized_tokens"),
        ),
        (
            "Source matmul vs quantized full A16W8",
            metrics.get("source_matmul_vs_quantized_full_a16w8_tokens"),
        ),
        (
            "Source matmul vs quantized residual-add A16W8",
            metrics.get("source_matmul_vs_quantized_residual_add_a16w8_tokens"),
        ),
        (
            "Source matmul vs quantized fallback QNNPACK",
            metrics.get("source_matmul_vs_quantized_fallback_qnnpack_tokens"),
        ),
        (
            "Source matmul vs quantized fallback TFLite pre-lowered",
            metrics.get("source_matmul_vs_quantized_fallback_tflite_pre_lowered_tokens"),
        ),
        (
            "Source matmul vs quantized fallback ai_edge_torch pre-lowered",
            metrics.get("source_matmul_vs_quantized_fallback_ai_edge_pre_lowered_tokens"),
        ),
        (
            "Source matmul vs quantized fallback TFLite A16W8 pre-lowered",
            metrics.get("source_matmul_vs_quantized_fallback_tflite_a16w8_pre_lowered_tokens"),
        ),
        (
            "Source matmul vs quantized fallback Apr 9 pre-lowered TFLite",
            metrics.get("source_matmul_vs_quantized_fallback_apr9_pre_lowered_tokens"),
        ),
        (
            "Source matmul vs quantized fallback Vela lowered board",
            metrics.get("source_matmul_vs_quantized_fallback_vela_lowered_board_tokens"),
        ),
        (
            "Fallback original non-quantized vs quantized fallback QNNPACK",
            metrics.get("fallback_original_non_quantized_vs_quantized_fallback_qnnpack_tokens"),
        ),
        (
            "Fallback original non-quantized vs quantized fallback TFLite pre-lowered",
            metrics.get("fallback_original_non_quantized_vs_quantized_fallback_tflite_pre_lowered_tokens"),
        ),
        (
            "Fallback original non-quantized vs quantized fallback ai_edge_torch pre-lowered",
            metrics.get("fallback_original_non_quantized_vs_quantized_fallback_ai_edge_pre_lowered_tokens"),
        ),
        (
            "Fallback original non-quantized vs quantized fallback TFLite A16W8 pre-lowered",
            metrics.get("fallback_original_non_quantized_vs_quantized_fallback_tflite_a16w8_pre_lowered_tokens"),
        ),
        (
            "Fallback original non-quantized vs quantized fallback Apr 9 pre-lowered TFLite",
            metrics.get("fallback_original_non_quantized_vs_quantized_fallback_apr9_pre_lowered_tokens"),
        ),
        (
            "Fallback original non-quantized vs quantized fallback Vela lowered board",
            metrics.get("fallback_original_non_quantized_vs_quantized_fallback_vela_lowered_board_tokens"),
        ),
        (
            "Quantized fallback QNNPACK vs TFLite pre-lowered",
            metrics.get("quantized_fallback_qnnpack_vs_tflite_pre_lowered_tokens"),
        ),
        (
            "Quantized fallback QNNPACK vs ai_edge_torch pre-lowered",
            metrics.get("quantized_fallback_qnnpack_vs_ai_edge_pre_lowered_tokens"),
        ),
        (
            "Quantized fallback QNNPACK vs TFLite A16W8 pre-lowered",
            metrics.get("quantized_fallback_qnnpack_vs_tflite_a16w8_pre_lowered_tokens"),
        ),
        (
            "Quantized fallback QNNPACK vs Apr 9 pre-lowered TFLite",
            metrics.get("quantized_fallback_qnnpack_vs_apr9_pre_lowered_tokens"),
        ),
        (
            "Quantized fallback QNNPACK vs Vela lowered board",
            metrics.get("quantized_fallback_qnnpack_vs_vela_lowered_board_tokens"),
        ),
    ]
    for label, row in token_rows:
        if not row:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    fmt(row["exact_fraction"]),
                    fmt(row["mismatch_count"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Runtime/Deployment",
            "",
            "| Model path | NPU ops | CPU fallback ops | Measured board runtime |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key, spec in CANDIDATES.items():
        if key not in wrappers:
            continue
        deployment = wrappers[key].get("deployment", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    spec["label"],
                    fmt(deployment.get("npu_ops", "n/a")),
                    fmt(deployment.get("cpu_fallback_ops", "n/a")),
                    fmt(deployment.get("measured_board_runtime_ms", "n/a")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `SSIM windowed` is an 11x11 Gaussian-window RGB SSIM averaged over channels.",
            "- `MS-SSIM` is computed over five image scales with standard MS-SSIM weights.",
            "- `LPIPS alex` is filled when the optional `lpips` package and AlexNet checkpoint are available.",
            "- p95/p99 error values are percentiles over absolute RGB pixel error in `[0, 1]`.",
            "- Full scalar details, including windowed SSIM, MSE/RMSE, max error, and per-channel error, are retained in `metrics.json`.",
            "- The source-matmul wrapper is built through existing project code: `build_encoder_quantizer_split(titok, encoder_variant=\"source_matmul_attention\")`.",
            "- The fallback model was loaded through the existing `load_fallback_model` path in `scripts/ptq/compare_fallback_vs_residual_add.py`.",
            "- Quantized pre-lowered rows reuse saved calibrated artifacts when present; calibration metadata is retained in `quantized_pre_lowered_run.json`.",
            "- Token agreement compares VQ token IDs. For fallback, these are the argmax token IDs from the fallback student logits.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update scalar metrics for the step-by-step TiTok comparison.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-image", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.setdefault("wrappers", {})
    metrics["image"] = str(args.input_image)
    metrics["image_size"] = args.image_size

    reference = load_rgb(args.input_image, args.image_size)
    for key, spec in CANDIDATES.items():
        candidate_path = output_dir / spec["image"]
        if not candidate_path.exists():
            continue
        wrapper = metrics["wrappers"].setdefault(key, {})
        wrapper["input_vs_reconstruction"] = image_metrics(reference, load_rgb(candidate_path, args.image_size))

    tokens = {key: load_tokens(output_dir / filename) for key, filename in TOKEN_FILES.items()}
    source_tokens = tokens.get("source_matmul_attention")
    fallback_tokens = tokens.get("fallback_original_non_quantized")
    quantized_full_tokens = tokens.get("quantized_full_a16w8")
    quantized_residual_tokens = tokens.get("quantized_residual_add_a16w8")
    quantized_fallback_tokens = tokens.get("quantized_fallback_qnnpack")
    quantized_fallback_tflite_tokens = tokens.get("quantized_fallback_tflite_pre_lowered")
    quantized_fallback_ai_edge_tokens = tokens.get("quantized_fallback_ai_edge_pre_lowered")
    quantized_fallback_tflite_a16w8_tokens = tokens.get("quantized_fallback_tflite_a16w8_pre_lowered")
    quantized_fallback_apr9_tokens = tokens.get("quantized_fallback_apr9_pre_lowered")
    quantized_fallback_vela_board_tokens = tokens.get("quantized_fallback_vela_lowered_board")

    for key in HIDDEN_WRAPPERS:
        metrics["wrappers"].pop(key, None)
    for key in (
        "default_vs_source_matmul_vq_tokens",
        "reconstruct_titok_example_s128_tokens_vs_default",
        "quantized_full_a16w8_vs_quantized_residual_add_a16w8_tokens",
        "quantized_residual_add_a16w8_vs_quantized_fallback_qnnpack_tokens",
    ):
        metrics.pop(key, None)
    if source_tokens is not None and fallback_tokens is not None:
        metrics["source_matmul_vs_fallback_original_non_quantized_tokens"] = token_agreement(
            source_tokens, fallback_tokens
        )
    if source_tokens is not None and quantized_full_tokens is not None:
        metrics["source_matmul_vs_quantized_full_a16w8_tokens"] = token_agreement(
            source_tokens, quantized_full_tokens
        )
    if source_tokens is not None and quantized_residual_tokens is not None:
        metrics["source_matmul_vs_quantized_residual_add_a16w8_tokens"] = token_agreement(
            source_tokens, quantized_residual_tokens
        )
    if source_tokens is not None and quantized_fallback_tokens is not None:
        metrics["source_matmul_vs_quantized_fallback_qnnpack_tokens"] = token_agreement(
            source_tokens, quantized_fallback_tokens
        )
    if source_tokens is not None and quantized_fallback_tflite_tokens is not None:
        metrics["source_matmul_vs_quantized_fallback_tflite_pre_lowered_tokens"] = token_agreement(
            source_tokens, quantized_fallback_tflite_tokens
        )
    if source_tokens is not None and quantized_fallback_ai_edge_tokens is not None:
        metrics["source_matmul_vs_quantized_fallback_ai_edge_pre_lowered_tokens"] = token_agreement(
            source_tokens, quantized_fallback_ai_edge_tokens
        )
    if source_tokens is not None and quantized_fallback_tflite_a16w8_tokens is not None:
        metrics["source_matmul_vs_quantized_fallback_tflite_a16w8_pre_lowered_tokens"] = token_agreement(
            source_tokens, quantized_fallback_tflite_a16w8_tokens
        )
    if source_tokens is not None and quantized_fallback_apr9_tokens is not None:
        metrics["source_matmul_vs_quantized_fallback_apr9_pre_lowered_tokens"] = token_agreement(
            source_tokens, quantized_fallback_apr9_tokens
        )
    if source_tokens is not None and quantized_fallback_vela_board_tokens is not None:
        metrics["source_matmul_vs_quantized_fallback_vela_lowered_board_tokens"] = token_agreement(
            source_tokens, quantized_fallback_vela_board_tokens
        )
    if fallback_tokens is not None and quantized_fallback_tokens is not None:
        metrics["fallback_original_non_quantized_vs_quantized_fallback_qnnpack_tokens"] = token_agreement(
            fallback_tokens, quantized_fallback_tokens
        )
    if fallback_tokens is not None and quantized_fallback_tflite_tokens is not None:
        metrics["fallback_original_non_quantized_vs_quantized_fallback_tflite_pre_lowered_tokens"] = token_agreement(
            fallback_tokens, quantized_fallback_tflite_tokens
        )
    if fallback_tokens is not None and quantized_fallback_ai_edge_tokens is not None:
        metrics["fallback_original_non_quantized_vs_quantized_fallback_ai_edge_pre_lowered_tokens"] = token_agreement(
            fallback_tokens, quantized_fallback_ai_edge_tokens
        )
    if fallback_tokens is not None and quantized_fallback_tflite_a16w8_tokens is not None:
        metrics["fallback_original_non_quantized_vs_quantized_fallback_tflite_a16w8_pre_lowered_tokens"] = token_agreement(
            fallback_tokens, quantized_fallback_tflite_a16w8_tokens
        )
    if fallback_tokens is not None and quantized_fallback_apr9_tokens is not None:
        metrics["fallback_original_non_quantized_vs_quantized_fallback_apr9_pre_lowered_tokens"] = token_agreement(
            fallback_tokens, quantized_fallback_apr9_tokens
        )
    if fallback_tokens is not None and quantized_fallback_vela_board_tokens is not None:
        metrics["fallback_original_non_quantized_vs_quantized_fallback_vela_lowered_board_tokens"] = token_agreement(
            fallback_tokens, quantized_fallback_vela_board_tokens
        )
    if quantized_fallback_tokens is not None and quantized_fallback_tflite_tokens is not None:
        metrics["quantized_fallback_qnnpack_vs_tflite_pre_lowered_tokens"] = token_agreement(
            quantized_fallback_tokens, quantized_fallback_tflite_tokens
        )
    if quantized_fallback_tokens is not None and quantized_fallback_ai_edge_tokens is not None:
        metrics["quantized_fallback_qnnpack_vs_ai_edge_pre_lowered_tokens"] = token_agreement(
            quantized_fallback_tokens, quantized_fallback_ai_edge_tokens
        )
    if quantized_fallback_tokens is not None and quantized_fallback_tflite_a16w8_tokens is not None:
        metrics["quantized_fallback_qnnpack_vs_tflite_a16w8_pre_lowered_tokens"] = token_agreement(
            quantized_fallback_tokens, quantized_fallback_tflite_a16w8_tokens
        )
    if quantized_fallback_tokens is not None and quantized_fallback_apr9_tokens is not None:
        metrics["quantized_fallback_qnnpack_vs_apr9_pre_lowered_tokens"] = token_agreement(
            quantized_fallback_tokens, quantized_fallback_apr9_tokens
        )
    if quantized_fallback_tokens is not None and quantized_fallback_vela_board_tokens is not None:
        metrics["quantized_fallback_qnnpack_vs_vela_lowered_board_tokens"] = token_agreement(
            quantized_fallback_tokens, quantized_fallback_vela_board_tokens
        )

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "step_by_step_metrics.md").write_text(markdown(metrics, output_dir))
    print(f"Updated {metrics_path}")
    print(f"Updated {output_dir / 'step_by_step_metrics.md'}")


if __name__ == "__main__":
    main()
