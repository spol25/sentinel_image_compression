# TiTok Deploy Tools

Standalone utilities for working with pretrained TiTok tokenizers without vendoring the upstream TiTok repository.

This project expects the upstream TiTok repository to be cloned separately. Pass its path with `--titok-root`.

## Layout

- `src/titok_deploy_tools/wrappers.py`: deployment-oriented wrapper modules
- `src/titok_deploy_tools/titok_env.py`: helper to load TiTok from an external checkout
- `scripts/reconstruct_titok_example.py`: reconstruct an image and save tokens
- `scripts/validate_titok_s128_wrapper.py`: validate the token-only wrapper against the original TiTok encode path

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

## Notes

- This repo does not vendor TiTok source or checkpoints.
- The scripts load pretrained checkpoints from Hugging Face using the TiTok code in the external checkout.
