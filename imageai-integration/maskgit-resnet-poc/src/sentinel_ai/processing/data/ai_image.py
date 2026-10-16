from time import time
from os import path
from turbojpeg import TurboJPEG, TJPF_BGR, TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE
import uuid
import cv2
from sentinel_shared_python import timing_utils

import logging

logger = logging.getLogger('ai_image')

cv2.setNumThreads(2)
_TJ = TurboJPEG()

class AIImage:
    def __init__(self, file_path=None, image=None, file_c=None, file_name=None):
        TARGET_WIDTH = 2000

        self.file_c = file_c
        self.file_p = int(time())
        self.file_path = file_path
        self.filename = file_name
        self.id = uuid.uuid4().hex
        if image is not None and file_c and file_name:
            self.image_data = image
        else:
            self.file_c = int(path.getmtime(file_path))
            self.filename = file_path.split('/')[-1]
            with open(file_path, "rb") as file:
                buffer = file.read()
            if buffer[:2] != b"\xff\xd8":
                import numpy as np
                self.image_data = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
                if self.image_data is None:
                    raise ValueError("Unsupported or corrupt input image")
            else:
                # Get image dimensions without decoding
                width, height, subsamp, colorspace = _TJ.decode_header(buffer)

                if width > TARGET_WIDTH * 4:
                    # Quarter scale
                    scaling = (1, 4)
                elif width > TARGET_WIDTH * 2:
                    # Half scale
                    scaling = (1, 2)
                else:
                    # Full scale
                    scaling = (1, 1)
                self.image_data = _TJ.decode(
                    buffer, pixel_format=TJPF_BGR,
                    scaling_factor=scaling,
                    flags=TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE
                )

    def is_image_valid(self, bbox):
        logger.info(str(bbox))
        h, w = self.image_data.shape[:2]
        left, top, right, bottom = [float(v) for v in bbox]
        return (abs(top - bottom) * h) > 5 and (abs(left - right) * w) > 5

    def save(self, file_path):
        with timing_utils.prof.timed("image-save"):
            ok, jpg_bytes = cv2.imencode(".jpg", self.image_data, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpg_bytes.tofile(file_path)

    def crop_from_bbox(self, bbox, expand=0):
        """
        Crops a region from an OpenCV image using a normalized bbox

        Inputs:
            image_buffer (np.ndarray): Input image in OpenCV (H, W, C) or (H, W)
            bbox (list): [x1, y1, x2, y2] in normalized [0,1] relative to the original image
            expand (float): Optional expansion factor (e.g., 0.05 = 5%)

        Returns:
            np.ndarray: Cropped image region
        """
        with timing_utils.prof.timed("image-crop"):
            h, w = self.image_data.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in bbox]

            # if expand > 0.0:
            #     cx = (x1 + x2) * 0.5
            #     cy = (y1 + y2) * 0.5
            #     w = (x2 - x1) * (1.0 + expand)
            #     h = (y2 - y1) * (1.0 + expand)
            #     x1, y1, x2, y2 = cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0

            logger.info(f"{x1}_{y1}_{x2}_{y2}_{h}_{w}")

            # Convert to pixel coords and clip
            px1 = max(0, min(int(x1 * w), w - 1))
            py1 = max(0, min(int(y1 * h), h - 1))
            px2 = max(0, min(int(x2 * w), w))
            py2 = max(0, min(int(y2 * h), h))

            # Ensure at least 1px
            if px2 <= px1: px2 = min(w, px1 + 1)
            if py2 <= py1: py2 = min(h, py1 + 1)

            # Crop with NumPy slicing
            image_crop = self.image_data[py1:py2, px1:px2]

            return AIImage(image=image_crop, file_c=self.file_c, file_name=self.filename)

    def draw_box(self, bbox, label, confidence, color=(0, 255, 0)):
        """
        Draws detections where bbox is mapped back to the ORIGINAL image size
        bbox format: [x1, y1, x2, y2], each in [0,1]
        Colors can be customized per class via class_colors (label -> (B,G,R))
        """
        with timing_utils.prof.timed("image-draw"):
            h, w = self.image_data.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in bbox]

            x1 = int(x1 * w)
            x2 = int(x2 * w)
            y1 = int(y1 * h)
            y2 = int(y2 * h)

            # Draw bounding box
            cv2.rectangle(self.image_data, (x1, y1), (x2, y2), color, 3)

            complete_label = f"{label} {confidence:.2f}"

            logger.info(complete_label)
            logger.info(confidence)

            # Measure text
            (tw, th), base = cv2.getTextSize(complete_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            y0 = max(0, y1 - th - 6)
            # Draw text box
            cv2.rectangle(self.image_data, (x1 - 2, y0), (x1 + tw + 8, y0 + th + 6), color, -1)
            # Draw label text
            cv2.putText(self.image_data, complete_label, (x1 + 4, y0 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            return self
