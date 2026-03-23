# TiTok-S-128 NXP/Arm Handoff Bundle

Files in this directory:
- `quantized_encoder.pt2`: PT2 artifact for the PTQ-converted encoder-only graph.
- `float_encoder_state_dict.pt`: pretrained float encoder weights.
- `latent_tokens.pt`: pretrained learned latent-token parameters used by the encoder.
- `vq_quantizer_state_dict.pt`: float TiTok VQ quantizer weights.
- `decoder_state_dict.pt`: TiTok decoder weights.
- `pixel_quantizer_state_dict.pt`: pixel-space quantizer used by the decode path when finetune_decoder is enabled.
- `pixel_decoder_state_dict.pt`: pixel-space decoder used by the decode path when finetune_decoder is enabled.
- `example_image.png`: example image included for quick testing.
- `titok_s128_config.json`: model config needed to reconstruct the TiTok modules.
- `minimal_titok_s128_handoff.py`: stripped down script showing how to load the bundle.
- `reconstruction_float_encoder.png`: saved reconstruction from the pretrained float encoder path.
- `reconstruction_quantized_encoder.png`: saved reconstruction from the quantized encoder PT2 path.

The encoder is the only quantized component in this bundle. VQ quantizer and decode path weights are saved in float form.

Run `minimal_titok_s128_handoff.py` with an optional first argument pointing to the recipient's `1d-tokenizer` checkout if it is not located next to this repo.

The bundle itself is intended for out-of-band sharing. The repo-side helper that generates it is `scripts/export/create_titok_s128_nxp_arm_handoff_bundle.py`.
