from sentinel_shared_python import path_utils, config_utils, timing_utils
import os
import logging
from .data.ai_config import AIConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger("ai-utils")


def get_all_available_models():
    file_map = {}
    # Match downloader names, and permit manually staged SD models.
    for directory in (path_utils.get_path_to_on_board_models(), path_utils.get_path_to_sd_models()):
        if os.path.isdir(directory):
            for name in sorted(os.listdir(directory)):
                full_path = os.path.join(directory, name)
                if name.endswith('.tflite') and os.path.isfile(full_path):
                    file_map.setdefault(os.path.splitext(name)[0], full_path)
    # Preserve compatibility with the original SD model filenames.
    for key, name in {'2_1': 'mdv51-tiny_vela.tflite', '6058_1': '6058_vela.tflite'}.items():
        full_path = os.path.join(path_utils.get_path_to_sd_models(), name)
        if key not in file_map and os.path.isfile(full_path):
            file_map[key] = full_path
    return file_map


def get_model_configs():
    model_config = config_utils.get_blob_config("model_settings")
    available_models = get_all_available_models()
    configs = {}
    if model_config and 'algs' in model_config:
        for config in model_config['algs']:
            logger.info(str(config))
            file_name = f"{config['alg_id']}_{config['version']}"
            if file_name in available_models:
                temp_config = AIConfig(config, available_models[file_name])
                configs[temp_config.get('alg_id')] = temp_config
        for config in configs.values():
            for downstream_config_alg_id in config.get_downstream_config_ids():
                config.downstream_configs.append(configs[downstream_config_alg_id])
    return configs.values()


def get_files(input_dir):
    with timing_utils.prof.timed("file-find"):
        files = []
        for file in sorted(os.listdir(input_dir)):
            path = os.path.join(input_dir, file)
            if not file.startswith(".") and os.path.isfile(path) and path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                files.append(path)
        return files
