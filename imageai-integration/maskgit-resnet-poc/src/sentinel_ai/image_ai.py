import os
import logging
import traceback
from .processing.ai_utils import get_model_configs, get_files
from .processing.data.ai_request import AIRequest
from sentinel_shared_python import timing_utils

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger("image-ai")


def execute():
    INPUT_DIR = "/mnt/sdcard/backlog"
    with timing_utils.prof.timed("total pipeline"):
        model_configs = list(get_model_configs())
        if not model_configs or not any(not c.is_intermediate() for c in model_configs):
            raise RuntimeError("No entry model configured; backlog retained")
        failures = []
        files = get_files(INPUT_DIR)
        file_count = len(files)

        for file_index, file in enumerate(files):
            succeeded = True
            for config_index, config in enumerate(model_configs):
                if not config.is_intermediate():
                    with timing_utils.prof.timed("per-image"):
                        logger.info(
                            f"Entry Model: "
                            f"Model: {config.get('alg_name')} v{config.get('version')} (ID: {config.get('alg_id')}, Framework: "
                            f"{config.get('framework')}) @ {config.get('sensitivity')} confidence"
                        )
                        logger.info(f'({file_index + 1}/{len(files)}) Processing ')
                        try:
                            responses = config.execute(AIRequest(file, config.get("alg_id")))
                            for response in responses:
                                response.write_insights()
                        except Exception as e:
                            succeeded = False
                            print(traceback.format_exc())
                            logger.error(f'Error: {e}')
            if succeeded:
                os.remove(file)
            else:
                failures.append(os.path.basename(file))
        if failures:
            raise RuntimeError(f"Processing failed; inputs retained: {failures}")
    timing_utils.prof.report(logger, num_images=file_count)
