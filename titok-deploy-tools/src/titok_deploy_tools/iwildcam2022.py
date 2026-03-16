import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

from PIL import Image


ANIMAL_CATEGORY = "1"


@dataclass
class CropCandidate:
    image_id: str
    image_path: str
    mask_path: str
    source_split: str
    detection_index: int
    detection_conf: float
    detection_category: str
    bbox_xywh_norm: list[float]
    bbox_xyxy_pixels: list[int]
    crop_area_pixels: int
    width: int
    height: int
    category_id: int | None = None
    location: str | None = None
    seq_id: str | None = None
    datetime: str | None = None


def load_json(path: Path):
    return json.loads(path.read_text())


def validate_iwildcam_layout(dataset_root: Path):
    required = [
        dataset_root / "metadata" / "iwildcam2022_mdv4_detections.json",
        dataset_root / "instance_masks",
        dataset_root / "train",
        dataset_root / "test",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"iWildCam 2022 dataset layout is incomplete: {joined}")


def build_train_category_lookup(dataset_root: Path) -> dict[str, int]:
    annotations_path = dataset_root / "metadata" / "iwildcam2022_train_annotations.json"
    if not annotations_path.exists():
        return {}
    annotations = load_json(annotations_path)
    image_to_category: dict[str, int] = {}
    for annotation in annotations.get("annotations", []):
        image_to_category[annotation["image_id"]] = annotation["category_id"]
    return image_to_category


def build_image_metadata_lookup(dataset_root: Path) -> dict[str, dict]:
    metadata_root = dataset_root / "metadata"
    image_lookup: dict[str, dict] = {}
    for metadata_path in sorted(metadata_root.glob("iwildcam2022_*.json")):
        try:
            payload = load_json(metadata_path)
        except json.JSONDecodeError:
            continue
        images = payload.get("images")
        if not isinstance(images, list):
            continue
        for image in images:
            image_id = image.get("id")
            file_name = image.get("file_name") or image.get("file")
            keys = set()
            if image_id is not None:
                keys.add(str(image_id))
            if file_name:
                keys.add(Path(file_name).stem)
            if not keys:
                continue
            record = {
                "location": image.get("location"),
                "seq_id": image.get("seq_id"),
                "datetime": image.get("datetime") or image.get("date_captured"),
                "file_name": file_name,
            }
            for key in keys:
                image_lookup[key] = {**image_lookup.get(key, {}), **{k: v for k, v in record.items() if v is not None}}
    return image_lookup


def clip_bbox_xyxy(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> list[int]:
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return [x0, y0, x1, y1]


def md_bbox_to_pixels(md_bbox: list[float], width: int, height: int) -> list[int]:
    x, y, w, h = md_bbox
    x0 = int(round(x * width))
    y0 = int(round(y * height))
    x1 = int(round((x + w) * width))
    y1 = int(round((y + h) * height))
    return clip_bbox_xyxy(x0, y0, x1, y1, width, height)


def mask_bbox_to_pixels(mask_path: Path, detection_index: int):
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("I")
        pixels = mask.load()
        width, height = mask.size
        target = detection_index + 1
        min_x = width
        min_y = height
        max_x = -1
        max_y = -1
        found = False
        for y in range(height):
            for x in range(width):
                if pixels[x, y] == target:
                    found = True
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
        if not found:
            return None
        return [min_x, min_y, max_x + 1, max_y + 1], width, height


def expand_bbox_xyxy(bbox_xyxy: list[int], width: int, height: int, margin_fraction: float) -> list[int]:
    x0, y0, x1, y1 = bbox_xyxy
    box_w = x1 - x0
    box_h = y1 - y0
    dx = int(round(box_w * margin_fraction))
    dy = int(round(box_h * margin_fraction))
    return clip_bbox_xyxy(x0 - dx, y0 - dy, x1 + dx, y1 + dy, width, height)


def iter_crop_candidates(
    dataset_root: Path,
    *,
    min_conf: float = 0.2,
    animal_only: bool = True,
    min_crop_size: int = 32,
    margin_fraction: float = 0.05,
):
    validate_iwildcam_layout(dataset_root)
    detections_path = dataset_root / "metadata" / "iwildcam2022_mdv4_detections.json"
    detections = load_json(detections_path)
    category_lookup = build_train_category_lookup(dataset_root)
    image_metadata_lookup = build_image_metadata_lookup(dataset_root)

    for image_record in detections.get("images", []):
        image_rel = image_record.get("file")
        if not image_rel:
            continue
        image_path = dataset_root / image_rel
        if not image_path.exists():
            continue
        image_id = Path(image_rel).stem
        mask_path = dataset_root / "instance_masks" / f"{image_id}.png"
        if not mask_path.exists():
            continue
        source_split = Path(image_rel).parts[0]
        image_metadata = image_metadata_lookup.get(image_id, {})
        detections_for_image = image_record.get("detections") or []

        with Image.open(image_path) as image:
            width, height = image.size

        for detection_index, detection in enumerate(detections_for_image):
            category = str(detection.get("category"))
            conf = float(detection.get("conf", 0.0))
            if conf < min_conf:
                continue
            if animal_only and category != ANIMAL_CATEGORY:
                continue
            bbox_norm = detection.get("bbox")
            if not bbox_norm or len(bbox_norm) != 4:
                continue

            mask_bbox = mask_bbox_to_pixels(mask_path, detection_index)
            if mask_bbox is None:
                bbox_xyxy = md_bbox_to_pixels(bbox_norm, width, height)
            else:
                bbox_xyxy, mask_width, mask_height = mask_bbox
                if mask_width != width or mask_height != height:
                    bbox_xyxy = md_bbox_to_pixels(bbox_norm, width, height)

            bbox_xyxy = expand_bbox_xyxy(bbox_xyxy, width, height, margin_fraction)
            crop_w = bbox_xyxy[2] - bbox_xyxy[0]
            crop_h = bbox_xyxy[3] - bbox_xyxy[1]
            if min(crop_w, crop_h) < min_crop_size:
                continue

            yield CropCandidate(
                image_id=image_id,
                image_path=str(image_path),
                mask_path=str(mask_path),
                source_split=source_split,
                detection_index=detection_index,
                detection_conf=conf,
                detection_category=category,
                bbox_xywh_norm=[float(v) for v in bbox_norm],
                bbox_xyxy_pixels=bbox_xyxy,
                crop_area_pixels=crop_w * crop_h,
                width=width,
                height=height,
                category_id=category_lookup.get(image_id),
                location=str(image_metadata.get("location")) if image_metadata.get("location") is not None else None,
                seq_id=str(image_metadata.get("seq_id")) if image_metadata.get("seq_id") is not None else None,
                datetime=str(image_metadata.get("datetime")) if image_metadata.get("datetime") is not None else None,
            )


def write_crop_index(output_path: Path, candidates: list[CropCandidate], metadata: dict):
    payload = {
        "metadata": metadata,
        "count": len(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def load_crop_index(index_path: Path) -> list[CropCandidate]:
    payload = load_json(index_path)
    return [CropCandidate(**candidate) for candidate in payload["candidates"]]


def select_unique_image_candidates(
    candidates: list[CropCandidate],
    *,
    max_crops_per_image: int = 1,
) -> list[CropCandidate]:
    grouped: dict[str, list[CropCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.image_id].append(candidate)

    selected: list[CropCandidate] = []
    for image_id in sorted(grouped):
        ranked = sorted(
            grouped[image_id],
            key=lambda candidate: (candidate.detection_conf, candidate.crop_area_pixels),
            reverse=True,
        )
        selected.extend(ranked[:max_crops_per_image])
    return selected


def sample_calibration_eval_candidates(
    candidates: list[CropCandidate],
    *,
    calibration_count: int,
    eval_count: int,
    seed: int,
):
    total_count = calibration_count + eval_count
    if len(candidates) < total_count:
        raise ValueError(
            f"Need at least {total_count} candidates, found {len(candidates)}."
        )
    labeled_candidates = [candidate for candidate in candidates if candidate.category_id is not None]
    pool = labeled_candidates if len(labeled_candidates) >= total_count else candidates
    rng = random.Random(seed)

    def choose_diverse_subset(available: list[CropCandidate], subset_count: int):
        available = list(available)
        rng.shuffle(available)
        selected: list[CropCandidate] = []
        category_counts: dict[int, int] = defaultdict(int)
        location_counts: dict[str, int] = defaultdict(int)
        used_seq_ids: set[str] = set()

        while len(selected) < subset_count and available:
            best_index = None
            best_score = None
            for index, candidate in enumerate(available):
                score = 0.25 * candidate.detection_conf
                score += 0.1 * min(candidate.crop_area_pixels / max(candidate.width * candidate.height, 1), 1.0)
                if candidate.category_id is not None:
                    score += 5.0 / (1 + category_counts[candidate.category_id])
                if candidate.location is not None:
                    score += 3.0 / (1 + location_counts[candidate.location])
                if candidate.seq_id is not None and candidate.seq_id not in used_seq_ids:
                    score += 2.0
                if candidate.source_split == "train":
                    score += 0.5
                score += rng.random() * 1e-6
                if best_score is None or score > best_score:
                    best_score = score
                    best_index = index

            chosen = available.pop(best_index)
            selected.append(chosen)
            if chosen.category_id is not None:
                category_counts[chosen.category_id] += 1
            if chosen.location is not None:
                location_counts[chosen.location] += 1
            if chosen.seq_id is not None:
                used_seq_ids.add(chosen.seq_id)

        if len(selected) != subset_count:
            raise ValueError(f"Unable to select {subset_count} diverse candidates from pool of {len(pool)}.")
        return selected

    eval_candidates = choose_diverse_subset(pool, eval_count)
    chosen_keys = {
        (candidate.image_id, candidate.detection_index)
        for candidate in eval_candidates
    }
    remaining = [
        candidate
        for candidate in pool
        if (candidate.image_id, candidate.detection_index) not in chosen_keys
    ]
    calibration_candidates = choose_diverse_subset(remaining, calibration_count)
    return calibration_candidates, eval_candidates


def export_crop(candidate: CropCandidate, output_path: Path):
    with Image.open(candidate.image_path) as image:
        crop = image.crop(tuple(candidate.bbox_xyxy_pixels))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)


def manifest_from_paths(paths: list[Path], *, source_manifest: Path | None = None, split: str | None = None):
    payload = {
        "count": len(paths),
        "images": [str(path) for path in paths],
    }
    if source_manifest is not None:
        payload["source_manifest"] = str(source_manifest)
    if split is not None:
        payload["split"] = split
    return payload
