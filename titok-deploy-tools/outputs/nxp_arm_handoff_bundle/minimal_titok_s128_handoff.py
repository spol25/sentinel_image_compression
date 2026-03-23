import json
import sys
from pathlib import Path

import torch
import torch.ao.quantization.fx._decomposed  # registers quantized_decomposed ops for pt2 load
from omegaconf import OmegaConf

BUNDLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BUNDLE_DIR.parents[1]
TITOK_ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO_ROOT.parent / '1d-tokenizer'

SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(TITOK_ROOT) not in sys.path:
    sys.path.insert(0, str(TITOK_ROOT))

from modeling.titok import TiTok
from titok_deploy_tools.ptq import build_encoder_quantizer_split
from titok_deploy_tools.utils import load_image, save_reconstruction

config = OmegaConf.create(json.loads((BUNDLE_DIR / 'titok_s128_config.json').read_text()))
titok = TiTok(config).eval().to('cpu')
titok.requires_grad_(False)
titok.encoder.load_state_dict(torch.load(BUNDLE_DIR / 'float_encoder_state_dict.pt', map_location='cpu'))
latent_tokens = torch.load(BUNDLE_DIR / 'latent_tokens.pt', map_location='cpu')
titok.latent_tokens.data.copy_(latent_tokens)
titok.quantize.load_state_dict(torch.load(BUNDLE_DIR / 'vq_quantizer_state_dict.pt', map_location='cpu'))
titok.decoder.load_state_dict(torch.load(BUNDLE_DIR / 'decoder_state_dict.pt', map_location='cpu'))
if hasattr(titok, 'pixel_quantize') and (BUNDLE_DIR / 'pixel_quantizer_state_dict.pt').exists():
    titok.pixel_quantize.load_state_dict(torch.load(BUNDLE_DIR / 'pixel_quantizer_state_dict.pt', map_location='cpu'))
if hasattr(titok, 'pixel_decoder') and (BUNDLE_DIR / 'pixel_decoder_state_dict.pt').exists():
    titok.pixel_decoder.load_state_dict(torch.load(BUNDLE_DIR / 'pixel_decoder_state_dict.pt', map_location='cpu'))

encoder_only, latents_to_tokens, _ = build_encoder_quantizer_split(titok)
quantized_encoder = torch.export.load(BUNDLE_DIR / 'quantized_encoder.pt2').module()

with torch.no_grad():
    image = load_image(BUNDLE_DIR / 'example_image.png', image_size=256)
    float_latent = encoder_only(image)
    float_tokens = latents_to_tokens(float_latent)
    float_reconstruction = titok.decode_tokens(float_tokens.unsqueeze(1) if float_tokens.ndim == 2 else float_tokens)

    latent = quantized_encoder(image)
    tokens = latents_to_tokens(latent)
    quantized_reconstruction = titok.decode_tokens(tokens.unsqueeze(1) if tokens.ndim == 2 else tokens)

save_reconstruction(float_reconstruction, BUNDLE_DIR / 'reconstruction_float_encoder.png')
save_reconstruction(quantized_reconstruction, BUNDLE_DIR / 'reconstruction_quantized_encoder.png')

print('tokens_shape:', tuple(tokens.shape))
print('tokens_sample:', tokens[0, :16].tolist() if tokens.ndim == 2 else tokens.reshape(tokens.shape[0], -1)[0, :16].tolist())
print('float_reconstruction_shape:', tuple(float_reconstruction.shape))
print('quantized_reconstruction_shape:', tuple(quantized_reconstruction.shape))
