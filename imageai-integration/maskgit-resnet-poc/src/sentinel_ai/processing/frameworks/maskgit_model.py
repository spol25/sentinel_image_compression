"""ImageAI image-to-token framework using the validated Ethos-U tokenizer."""
import logging
import numpy as np
from PIL import Image
from .token_encoder import TokenEncoder
from ..data.ai_token_response import AITokenResponse
from sentinel_shared_python import timing_utils

logger = logging.getLogger("maskgit")

class TokenFramework:
    kind = "maskgit"
    def __init__(self, framework_id, config):
        if config.has_downstream_configs() or config.get_downstream_config_ids():
            raise ValueError("Token output cannot feed detection/classification stages")
        self.config = config
        self.encoder = TokenEncoder(config.get_model_file(), kind=self.kind)
        logger.info("%s ready: Ethos-U delegate, model=%s", self.kind, self.encoder.model_sha256)

    def execute(self, request):
        request.load()
        with timing_utils.prof.timed(self.kind + "-preprocess"):
            rgb = request.image.image_data[:, :, ::-1]
            image = Image.fromarray(rgb).resize((256, 256), Image.Resampling.BICUBIC)
            canonical = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / np.float32(255)
            q, prep = self.encoder.prepare(canonical)
        with timing_utils.prof.timed(self.kind + "-inference"):
            payload, _, stats = self.encoder.run_prepared(q)
        return AITokenResponse(request, self.config, self.encoder.model_sha256, payload, {**prep, **stats}, self.kind)


class MaskGITFramework(TokenFramework):
    kind = "maskgit"


class ResNetTokenizerFramework(TokenFramework):
    kind = "resnet"
