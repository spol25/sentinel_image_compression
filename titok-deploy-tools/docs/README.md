# TiTok Deploy Tools

Standalone utilities for working with pretrained TiTok tokenizers without vendoring the upstream TiTok repository.

This project expects its model and deployment dependencies to be cloned
separately. Pass the TiTok checkout with `--titok-root`.

## Required Repositories

Use these forks and pinned commits for the reproducible UCM-i.MX93 + Ethos-U65
workflow. The upstream repositories do not contain all required changes.

| Purpose | Repository | Branch | Pinned commit |
| --- | --- | --- | --- |
| TiTok model and patched attention helpers | [`spol25/1d-tokenizer`](https://github.com/spol25/1d-tokenizer) | `main` | `ba028d08fbce1c7a03f3661b7f1e17b54c03548f` |
| ExecuTorch portable kernel-only build support | [`spol25/executorch`](https://github.com/spol25/executorch) | `main` | `dd873c3e8ccd1d5b0af5693a583d69b8d9ab5bc3` |
| UCM-i.MX93 CM33 runner | [`spol25/Executorch_runner_cm33`](https://github.com/spol25/Executorch_runner_cm33) | `main` | `1842b9b28d014e8fcaf780f7fab193cc4dabe247` |

The pinned TiTok commit includes both the encoder attention-layout rewrite and
the chunked BHLD attention helpers in `modeling/modules/blocks.py`.

## Layout

- `src/titok_deploy_tools/wrapper_tools/wrappers.py`: deployment-oriented wrapper modules
- `src/titok_deploy_tools/wrapper_tools/decode.py`: cloud-side token decode helpers
- `src/titok_deploy_tools/wrapper_tools/titok_env.py`: helper to load TiTok from an external checkout
- `src/titok_deploy_tools/wrapper_tools/utils.py`: shared utility helpers
- `src/titok_deploy_tools/ptq_tools/`: PTQ helpers
- `src/titok_deploy_tools/lowering_tools/`: ExecuTorch and Ethos-U lowering helpers
- `src/titok_deploy_tools/board_tools/`: CM33 input/output and board artifact helpers
- `docs/RUNBOOK.md`: end-to-end PTQ, lowering, board loading, remoteproc, fallback, and patch handoff
- `docs/cm33_legacy.md`: legacy CM33/UCM-i.MX93 bring-up log and historical context
- `scripts/reconstruct_titok_example.py`: reconstruct an image and save tokens
- `scripts/validate_titok_s128_wrapper.py`: validate the token-only wrapper against the original TiTok encode path
- `scripts/validate_decode_titok_tokens.py`: validate `decode.py` using saved wrapper tokens
- `scripts/export_and_lower/`: export and lowering scripts
- `scripts/ptq/`: PTQ preparation scripts

## Setup

1. Clone and pin the required repositories:

```bash
git clone https://github.com/spol25/1d-tokenizer.git
git -C 1d-tokenizer checkout ba028d08fbce1c7a03f3661b7f1e17b54c03548f

git clone https://github.com/spol25/executorch.git executorch-main
git -C executorch-main checkout dd873c3e8ccd1d5b0af5693a583d69b8d9ab5bc3

git clone https://github.com/spol25/Executorch_runner_cm33.git
git -C Executorch_runner_cm33 checkout 1842b9b28d014e8fcaf780f7fab193cc4dabe247
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
python scripts/export_and_lower/export_titok_s128_wrapper.py \
  --titok-root /path/to/1d-tokenizer
```

```bash
python scripts/export_and_lower/export_executorch_titok_s128_wrapper.py
```

```bash
python scripts/export_and_lower/validate_pte_titok_s128_wrapper.py \
  --titok-root /path/to/1d-tokenizer
```

```bash
python scripts/export_and_lower/lower_ethosu_titok_s128_encoder.py \
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
