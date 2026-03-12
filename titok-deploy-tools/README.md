# TiTok Deploy Tools

Standalone utilities for working with pretrained TiTok tokenizers without vendoring the upstream TiTok repository.

This project expects the upstream TiTok repository to be cloned separately. Pass its path with `--titok-root`.

## Layout

- `src/titok_deploy_tools/wrappers.py`: deployment-oriented wrapper modules
- `src/titok_deploy_tools/decode.py`: cloud-side token decode helpers
- `src/titok_deploy_tools/titok_env.py`: helper to load TiTok from an external checkout
- `scripts/reconstruct_titok_example.py`: reconstruct an image and save tokens
- `scripts/validate_titok_s128_wrapper.py`: validate the token-only wrapper against the original TiTok encode path
- `scripts/decode_titok_tokens.py`: decode transmitted token IDs into an image
- `scripts/validate_s128_roundtrip.py`: validate wrapper-token round-trip decode
- `scripts/export_titok_s128_wrapper.py`: export the S-128 token wrapper with `torch.export`
- `scripts/build_calibration_manifest.py`: build a representative PTQ calibration manifest
- `scripts/run_s128_calibration_baseline.py`: record baseline wrapper tokens for the calibration set

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
  --titok-root /path/to/1d-tokenizer
```

```bash
python scripts/decode_titok_tokens.py \
  --titok-root /path/to/1d-tokenizer \
  --repo-id yucornetto/tokenizer_titok_s128_imagenet \
  --tokens-json outputs/standalone_s128_tokens.json
```

```bash
python scripts/export_titok_s128_wrapper.py \
  --titok-root /path/to/1d-tokenizer
```

```bash
python scripts/build_calibration_manifest.py \
  --image-dir /path/to/representative/images \
  --output-dir outputs/ptq
```

```bash
python scripts/run_s128_calibration_baseline.py \
  --titok-root /path/to/1d-tokenizer \
  --manifest outputs/ptq/calibration_manifest.json \
  --output-dir outputs/ptq
```

## PTQ Prep

Before PTQ, prepare three things:

1. A representative calibration image set at the same image distribution you expect on device.
2. A calibration manifest listing the exact images used.
3. Baseline wrapper token outputs from the float model on that calibration set.

The acceptance checks for PTQ should be based on:

- token agreement rate between float and PTQ wrappers
- reconstruction quality after decoding PTQ tokens in the cloud
- failure-case inspection for images whose token assignments change substantially

## Notes

- This repo does not vendor TiTok source or checkpoints.
- The scripts load pretrained checkpoints from Hugging Face using the TiTok code in the external checkout.
