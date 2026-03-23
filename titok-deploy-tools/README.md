# TiTok Deploy Tools

Standalone utilities for working with pretrained TiTok tokenizers without vendoring the upstream TiTok repository.

This project expects the upstream TiTok repository to be cloned separately. Pass its path with `--titok-root`.

## Layout

- `src/titok_deploy_tools/wrappers.py`: deployment-oriented wrapper modules
- `src/titok_deploy_tools/decode.py`: cloud-side token decode helpers
- `src/titok_deploy_tools/titok_env.py`: helper to load TiTok from an external checkout
- `src/titok_deploy_tools/utils.py`: shared utility helpers
- `scripts/reconstruct_titok_example.py`: reconstruct an image and save tokens
- `scripts/validate_titok_s128_wrapper.py`: validate the token-only wrapper against the original TiTok encode path
- `scripts/validate_decode_titok_tokens.py`: validate `decode.py` using saved wrapper tokens
- `scripts/export/`: export-only scripts
- `scripts/ptq/`: PTQ preparation scripts

## Setup

1. Clone TiTok separately:

```bash
git clone https://github.com/bytedance/1d-tokenizer.git
```

2. Install TiTok dependencies in your environment.

3. Run these scripts with `--titok-root`:

```bash
python scripts/reconstruct_titok_example.py \
  --titok-root /path/to/1d-tokenizer \
  --output-dir outputs
```

```bash
python scripts/validate_titok_s128_wrapper.py \
  --titok-root /path/to/1d-tokenizer \
  --output-dir outputs \
  --tokens-output s128_wrapper_tokens.json
```

```bash
python scripts/validate_decode_titok_tokens.py \
  --titok-root /path/to/1d-tokenizer \
  --repo-id yucornetto/tokenizer_titok_s128_imagenet \
  --output-dir outputs \
  --tokens-json s128_wrapper_tokens.json
```

```bash
python scripts/export/export_titok_s128_wrapper.py \
  --titok-root /path/to/1d-tokenizer
```

```bash
python scripts/export/export_executorch_titok_s128_wrapper.py
```

```bash
python scripts/export/validate_pte_titok_s128_wrapper.py \
  --titok-root /path/to/1d-tokenizer
```

```bash
python scripts/export/lower_ethosu_titok_s128_encoder.py \
  --titok-root /path/to/1d-tokenizer \
  --manifest /path/to/calibration_manifest.json \
  --per-channel
```

```bash
python scripts/ptq/prepare_image_manifest.py \
  --image-dir /path/to/representative/images \
  --output-dir outputs/ptq
```

```bash
python scripts/ptq/split_calibration_eval_manifests.py \
  --manifest outputs/ptq/image_manifest.json \
  --output-dir outputs/ptq \
  --eval-count 32 \
  --shuffle \
  --seed 0
```

```bash
python scripts/ptq/run_s128_calibration_baseline.py \
  --titok-root /path/to/1d-tokenizer \
  --manifest outputs/ptq/calibration_manifest.json \
  --output-dir outputs/ptq
```

```bash
python scripts/ptq/run_encoder_ptq_experiment.py \
  --titok-root /path/to/1d-tokenizer \
  --manifest outputs/ptq/calibration_manifest.json \
  --output-dir outputs/ptq \
  --per-channel
```

```bash
python scripts/ptq/compare_token_outputs.py \
  --reference outputs/ptq/s128_float_baseline_tokens.json \
  --candidate outputs/ptq/s128_encoder_ptq_tokens.json \
  --output-dir outputs/ptq
```

```bash
python scripts/ptq/compare_decoded_reconstructions.py \
  --titok-root /path/to/1d-tokenizer \
  --reference outputs/ptq/s128_float_baseline_tokens.json \
  --candidate outputs/ptq/s128_encoder_ptq_tokens.json \
  --output-dir outputs/ptq
```

## PTQ Prep

Before PTQ, prepare three things:

1. A representative calibration image set at the same image distribution you expect on device.
2. A generic image manifest listing the full candidate image pool.
3. Separate non-overlapping calibration and eval manifests derived from that pool.
4. Baseline wrapper token outputs from the float model on the calibration set.

The eval manifest should stay separate from calibration so token-agreement and reconstruction metrics are not measured on the same images used for observer calibration.

The current PTQ path is intentionally split at the TiTok encoder/VQ boundary:

- export the encoder-only boundary with `torch.export`
- apply PTQ preparation and conversion to that exported encoder graph
- keep the TiTok VQ quantizer unquantized and run it in float
- compare final token IDs and decoded reconstructions against the float baseline

The acceptance checks for PTQ should be based on:

- token agreement rate between float and PTQ wrappers
- reconstruction quality after decoding PTQ tokens in the cloud
- failure-case inspection for images whose token assignments change substantially

## Notes

- This repo does not vendor TiTok source or checkpoints.
- The scripts load pretrained checkpoints from Hugging Face using the TiTok code in the external checkout.
