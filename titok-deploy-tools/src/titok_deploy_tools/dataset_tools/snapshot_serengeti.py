import json
import random
import subprocess
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import ijson
from PIL import Image


AZURE_BASE_URL = "https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-unzipped"


@dataclass
class SerengetiCropCandidate:
    image_id: str
    file_name: str
    bbox_xywh_pixels: list[float]
    bbox_xyxy_pixels: list[int]
    bbox_area_pixels: float
    category_id: int | None
    category_name: str | None
    location: str | None
    seq_id: str | None
    datetime: str | None
    season: str | None


@dataclass
class QuickSerengetiCropCandidate:
    image_id: str
    file_name: str
    bbox_xywh_pixels: list[float]
    bbox_area_pixels: float
    location: str | None
    camera_id: str | None
    season: str | None


def load_json(path: Path):
    return json.loads(path.read_text())


def stream_categories(metadata_path: Path) -> dict[int, str]:
    categories: dict[int, str] = {}
    with metadata_path.open("rb") as handle:
        for category in ijson.items(handle, "categories.item"):
            categories[int(category["id"])] = category["name"]
    return categories


def normalize_image_key(value) -> str | None:
    if value is None:
        return None
    return str(value)


def build_image_lookup(metadata_path: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    with metadata_path.open("rb") as handle:
        for image in ijson.items(handle, "images.item"):
            keys = {
                normalize_image_key(image.get("id")),
                normalize_image_key(image.get("seq_id")),
                Path(image.get("file_name", "")).stem if image.get("file_name") else None,
            }
            record = {
                "id": normalize_image_key(image.get("id")),
                "file_name": image.get("file_name"),
                "location": image.get("location"),
                "seq_id": normalize_image_key(image.get("seq_id")),
                "datetime": image.get("datetime") or image.get("date_captured"),
                "season": normalize_image_key(image.get("season")),
                "width": image.get("width"),
                "height": image.get("height"),
            }
            for key in keys:
                if key:
                    lookup[key] = {**lookup.get(key, {}), **{k: v for k, v in record.items() if v is not None}}
    return lookup


def build_species_annotation_lookup(metadata_path: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    with metadata_path.open("rb") as handle:
        for annotation in ijson.items(handle, "annotations.item"):
            image_id = normalize_image_key(annotation.get("image_id"))
            if not image_id:
                continue
            record = {
                "species_category_id": int(annotation["category_id"]) if annotation.get("category_id") is not None else None,
                "location": normalize_image_key(annotation.get("location")),
                "seq_id": normalize_image_key(annotation.get("seq_id")),
                "datetime": annotation.get("datetime"),
                "season": normalize_image_key(annotation.get("season")),
                "sequence_level_annotation": annotation.get("sequence_level_annotation"),
            }
            lookup[image_id] = {**lookup.get(image_id, {}), **{k: v for k, v in record.items() if v is not None}}
    return lookup


def clip_bbox_xyxy(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> list[int]:
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return [x0, y0, x1, y1]


def bbox_xywh_to_xyxy(bbox_xywh: list[float], width: int, height: int, margin_fraction: float) -> list[int]:
    x, y, w, h = bbox_xywh
    dx = w * margin_fraction
    dy = h * margin_fraction
    x0 = int(round(x - dx))
    y0 = int(round(y - dy))
    x1 = int(round(x + w + dx))
    y1 = int(round(y + h + dy))
    return clip_bbox_xyxy(x0, y0, x1, y1, width, height)


def image_id_to_file_name(image_id: str) -> str:
    return f"{image_id}.JPG"


def infer_dimensions(annotation: dict, image_record: dict):
    width = image_record.get("width")
    height = image_record.get("height")
    bbox_xywh = annotation.get("bbox")
    if width is None or height is None:
        if not bbox_xywh or len(bbox_xywh) != 4:
            return None, None
        width = max(int(round(bbox_xywh[0] + bbox_xywh[2])), 1)
        height = max(int(round(bbox_xywh[1] + bbox_xywh[3])), 1)
    return int(width), int(height)


def iter_bbox_candidates(
    metadata_path: Path,
    bbox_path: Path,
    *,
    min_bbox_size: int = 32,
    margin_fraction: float = 0.05,
):
    image_lookup = build_image_lookup(metadata_path)
    species_lookup = build_species_annotation_lookup(metadata_path)
    category_lookup = stream_categories(metadata_path)

    bbox_metadata = load_json(bbox_path)
    for annotation in bbox_metadata.get("annotations", []):
        bbox_xywh = annotation.get("bbox")
        if not bbox_xywh or len(bbox_xywh) != 4:
            continue
        image_key = normalize_image_key(annotation.get("image_id"))
        image_record = image_lookup.get(image_key)
        if image_record is None:
            continue
        file_name = image_record.get("file_name")
        if not file_name:
            continue

        width, height = infer_dimensions(annotation, image_record)
        if width is None or height is None:
            continue

        species_record = species_lookup.get(image_key, {})

        bbox_xyxy = bbox_xywh_to_xyxy(bbox_xywh, width, height, margin_fraction)
        crop_w = bbox_xyxy[2] - bbox_xyxy[0]
        crop_h = bbox_xyxy[3] - bbox_xyxy[1]
        if min(crop_w, crop_h) < min_bbox_size:
            continue

        category_id = annotation.get("category_id")
        if category_id is not None:
            category_id = int(category_id)

        yield SerengetiCropCandidate(
            image_id=image_record.get("id") or Path(file_name).stem,
            file_name=file_name,
            bbox_xywh_pixels=[float(v) for v in bbox_xywh],
            bbox_xyxy_pixels=bbox_xyxy,
            bbox_area_pixels=float(bbox_xywh[2] * bbox_xywh[3]),
            category_id=species_record.get("species_category_id"),
            category_name=category_lookup.get(species_record.get("species_category_id")),
            location=species_record.get("location") or normalize_image_key(image_record.get("location")),
            seq_id=species_record.get("seq_id") or normalize_image_key(image_record.get("seq_id")),
            datetime=species_record.get("datetime") or image_record.get("datetime"),
            season=species_record.get("season") or normalize_image_key(image_record.get("season")),
        )


def iter_quick_animal_candidates(bbox_path: Path):
    bbox_metadata = load_json(bbox_path)
    for annotation in bbox_metadata.get("annotations", []):
        if int(annotation.get("category_id", -1)) != 1:
            continue
        image_id = normalize_image_key(annotation.get("image_id"))
        bbox_xywh = annotation.get("bbox")
        if not image_id or not bbox_xywh or len(bbox_xywh) != 4:
            continue
        parts = Path(image_id).parts
        season = parts[0] if len(parts) > 0 else None
        location = parts[1] if len(parts) > 1 else None
        camera_id = parts[2] if len(parts) > 2 else None
        yield QuickSerengetiCropCandidate(
            image_id=image_id,
            file_name=image_id_to_file_name(image_id),
            bbox_xywh_pixels=[float(v) for v in bbox_xywh],
            bbox_area_pixels=float(bbox_xywh[2] * bbox_xywh[3]),
            location=location,
            camera_id=camera_id,
            season=season,
        )


def write_candidate_index(output_path: Path, candidates: list[SerengetiCropCandidate], metadata: dict):
    payload = {
        "metadata": metadata,
        "count": len(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def load_candidate_index(index_path: Path) -> list[SerengetiCropCandidate]:
    payload = load_json(index_path)
    return [SerengetiCropCandidate(**candidate) for candidate in payload["candidates"]]


def choose_best_box_per_image(candidates: list[SerengetiCropCandidate]) -> list[SerengetiCropCandidate]:
    grouped: dict[str, list[SerengetiCropCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.file_name].append(candidate)
    selected: list[SerengetiCropCandidate] = []
    for file_name in sorted(grouped):
        ranked = sorted(
            grouped[file_name],
            key=lambda candidate: (candidate.bbox_area_pixels, candidate.category_id is not None),
            reverse=True,
        )
        selected.append(ranked[0])
    return selected


def choose_best_quick_box_per_image(candidates: list[QuickSerengetiCropCandidate]) -> list[QuickSerengetiCropCandidate]:
    grouped: dict[str, list[QuickSerengetiCropCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.file_name].append(candidate)
    selected: list[QuickSerengetiCropCandidate] = []
    for file_name in sorted(grouped):
        ranked = sorted(grouped[file_name], key=lambda candidate: candidate.bbox_area_pixels, reverse=True)
        selected.append(ranked[0])
    return selected


def select_representative_split(
    candidates: list[SerengetiCropCandidate],
    *,
    calibration_count: int,
    eval_count: int,
    seed: int,
):
    total_count = calibration_count + eval_count
    if len(candidates) < total_count:
        raise ValueError(f"Need at least {total_count} candidates, found {len(candidates)}.")

    rng = random.Random(seed)
    pool = list(candidates)
    rng.shuffle(pool)

    def score_candidate(candidate, category_counts, location_counts, season_counts, used_seq_ids):
        score = 0.0
        score += min(candidate.bbox_area_pixels / 50000.0, 4.0)
        if candidate.category_id is not None:
            score += 8.0 / (1 + category_counts[candidate.category_id])
        if candidate.location is not None:
            score += 4.0 / (1 + location_counts[candidate.location])
        if candidate.season is not None:
            score += 2.0 / (1 + season_counts[candidate.season])
        if candidate.seq_id is not None and candidate.seq_id not in used_seq_ids:
            score += 2.0
        score += rng.random() * 1e-6
        return score

    def pick_subset(available, subset_count):
        selected = []
        category_counts = Counter()
        location_counts = Counter()
        season_counts = Counter()
        used_seq_ids = set()
        available = list(available)
        while len(selected) < subset_count and available:
            best_index = None
            best_score = None
            for index, candidate in enumerate(available):
                score = score_candidate(candidate, category_counts, location_counts, season_counts, used_seq_ids)
                if best_score is None or score > best_score:
                    best_score = score
                    best_index = index
            chosen = available.pop(best_index)
            selected.append(chosen)
            if chosen.category_id is not None:
                category_counts[chosen.category_id] += 1
            if chosen.location is not None:
                location_counts[chosen.location] += 1
            if chosen.season is not None:
                season_counts[chosen.season] += 1
            if chosen.seq_id is not None:
                used_seq_ids.add(chosen.seq_id)
        if len(selected) != subset_count:
            raise ValueError(f"Unable to select {subset_count} candidates.")
        return selected

    eval_candidates = pick_subset(pool, eval_count)
    chosen_files = {candidate.file_name for candidate in eval_candidates}
    remaining = [candidate for candidate in pool if candidate.file_name not in chosen_files]
    calibration_candidates = pick_subset(remaining, calibration_count)
    return calibration_candidates, eval_candidates


def build_remote_image_url(file_name: str, base_url: str = AZURE_BASE_URL) -> str:
    encoded_parts = [urllib.parse.quote(part) for part in Path(file_name).parts]
    return f"{base_url.rstrip('/')}/{'/'.join(encoded_parts)}"


def download_image(file_name: str, output_path: Path, base_url: str = AZURE_BASE_URL):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    candidate_names = [file_name]
    if file_name.endswith(".JPG"):
        candidate_names.append(file_name[:-4] + ".jpg")
    for candidate_name in candidate_names:
        url = build_remote_image_url(candidate_name, base_url=base_url)
        try:
            subprocess.run(
                ["curl", "-fL", "-sS", "-o", str(output_path), url],
                check=True,
            )
            return
        except Exception:
            if output_path.exists():
                output_path.unlink()
    raise FileNotFoundError(f"Unable to download {file_name} from {base_url}")


def export_crop(source_image_path: Path, bbox_xyxy_pixels: list[int], output_path: Path):
    with Image.open(source_image_path) as image:
        crop = image.crop(tuple(bbox_xyxy_pixels))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)


def export_crop_from_xywh(source_image_path: Path, bbox_xywh_pixels: list[float], output_path: Path, margin_fraction: float = 0.05):
    with Image.open(source_image_path) as image:
        width, height = image.size
        bbox_xyxy = bbox_xywh_to_xyxy(bbox_xywh_pixels, width, height, margin_fraction)
        crop = image.crop(tuple(bbox_xyxy))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)


def manifest_payload(paths: list[Path], *, source_manifest: Path | None = None, split: str | None = None):
    payload = {
        "count": len(paths),
        "images": [str(path) for path in paths],
    }
    if source_manifest is not None:
        payload["source_manifest"] = str(source_manifest)
    if split is not None:
        payload["split"] = split
    return payload
