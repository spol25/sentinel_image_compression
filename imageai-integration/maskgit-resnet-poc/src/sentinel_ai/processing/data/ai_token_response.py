"""Durable token results in ImageAI's data database; no reconstruction/export."""
import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from sentinel_shared_python import path_utils

logger = logging.getLogger("token-result")

class AITokenResponse:
    def __init__(self, request, config, model_sha, payload, metrics, kind="maskgit"):
        self.kind = kind
        self.count, self.codebook = {"maskgit": (256, 1024), "resnet": (128, 4096)}[kind]
        self.request, self.config = request, config
        self.model_sha, self.payload, self.metrics = model_sha, payload, metrics

    def save_data(self, save_bb, save_cropped):
        if save_bb or save_cropped:
            raise ValueError("Token models do not produce bounding boxes or cropped images")

    def write_insights(self):
        if len(self.payload) != 2 * self.count:
            raise ValueError(f"{self.kind} output must contain {self.count} little-endian uint16 tokens")
        source_sha = hashlib.sha256(Path(self.request.file_path).read_bytes()).hexdigest()
        model_id = f"{self.config.get('alg_id')}_{self.config.get('version')}"
        identity = f"{model_id}:{self.model_sha}:{source_sha}:rgb-u8-bicubic-256-unit-v1"
        result_id = hashlib.sha256(identity.encode()).hexdigest()
        with sqlite3.connect(path_utils.get_path_to_db()) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS token_outputs (
                result_id TEXT PRIMARY KEY, model_id TEXT NOT NULL, model_sha256 TEXT NOT NULL,
                original_image_name TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                created_at INTEGER NOT NULL, codec TEXT NOT NULL, token_count INTEGER NOT NULL,
                token_dtype TEXT NOT NULL, token_grid TEXT NOT NULL, codebook_size INTEGER NOT NULL,
                preprocess TEXT NOT NULL, payload BLOB NOT NULL, payload_sha256 TEXT NOT NULL,
                metrics_json TEXT NOT NULL, delivery_status TEXT NOT NULL DEFAULT 'pending')""")
            db.execute("""INSERT OR IGNORE INTO token_outputs
                (result_id,model_id,model_sha256,original_image_name,source_sha256,created_at,
                 codec,token_count,token_dtype,token_grid,codebook_size,preprocess,payload,
                 payload_sha256,metrics_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (result_id, model_id, self.model_sha, self.request.original_image_name, source_sha,
                 int(time.time()), self.kind + '-vq-v1', self.count, 'uint16-le', '[16,16]' if self.kind == 'maskgit' else '[128]', self.codebook,
                 'rgb-u8-bicubic-256-unit-v1', self.payload,
                 hashlib.sha256(self.payload).hexdigest(), json.dumps(self.metrics, sort_keys=True)))
        logger.info("Token result saved: image=%s id=%s tokens=%s bytes=%s delivery=pending",
                    self.request.original_image_name, result_id, self.count, len(self.payload))
