# ImageAI MaskGIT and ResNet proof of concept

This directory preserves the exact ImageAI integration that ran successfully on the Sentinel V2 i.MX93 board on October 12–13, 2026. It is a proof-of-concept snapshot, not the proposed production pipeline.

## Proven behavior

- MaskGIT ran through the installed ImageAI launcher and produced 256 little-endian uint16 token IDs (512 bytes) for each of six images.
- ResNet ran through the same launcher and produced 128 little-endian uint16 token IDs (256 bytes) for each image.
- All 1,536 MaskGIT token IDs and all 768 ResNet token IDs matched the previously validated board captures exactly.
- Existing local decoders reconstructed the outputs.
- Results were stored idempotently in the SD card's `local.db` table `token_outputs`.

## Provenance

The patch was prepared against `cxl-garage/sentinel-shared-python` commit `ae1bd252338159e46c9bdccf5cc37455bf7ed1e4`. The `src` directory contains exact snapshots of the seven changed or added files so the working implementation remains recoverable even if the patch no longer applies cleanly.

Validated encoder artifact hashes:

- MaskGIT: `0a418626321b95b96125d7aa61a07ce27f453cb06d042f23f146b9f59a3c4b2b`
- ResNet: `561ecd5aa6408fb72f48ae556654ad4a5b111b0e0a9d1979e1e17bc2e8cff25b`

## Known limitations

- Each tokenizer was selected as a standalone ImageAI entry model and processed the full backlog image.
- The code does not implement the production YOLO -> classifier -> crop -> compressor flow.
- It does not implement automatic MaskGIT-to-ResNet fallback.
- Model settings were edited locally and models were staged manually.
- Token rows remain `delivery_status='pending'`; no uploader or cloud decoder service is included.
- Model hashes are embedded in code.
- Runtime table creation is suitable for preserving the proof of concept but should be replaced by an agreed production database migration.

The production integration is being developed separately so this known-working snapshot remains unchanged.
