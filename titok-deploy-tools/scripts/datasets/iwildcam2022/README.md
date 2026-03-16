# iWildCam 2022 PTQ Crop Prep

This folder contains the source-specific code for turning the iWildCam 2022 dataset into a PTQ-ready crop dataset for TiTok.

The workflow assumes:

- you have already downloaded and extracted the iWildCam 2022 dataset locally
- the extracted tree contains:
  - `metadata/iwildcam2022_mdv4_detections.json`
  - `instance_masks/`
  - `train/`
  - `test/`
- crops and manifests should be written to an external disk, by default `/Volumes/Media/iwildcam2022_ptq`

The crop selection logic tries to be representative by using metadata when available:

- prefers labeled train images when enough are available
- diversifies across `category_id` where present
- diversifies across `location`
- avoids reusing the same `seq_id` when that metadata is available
- keeps at most one crop per source image by default

## 1. Build the crop index

```bash
python scripts/datasets/iwildcam2022/build_crop_index.py \
  --dataset-root /path/to/iwildcam2022 \
  --output /Volumes/Media/iwildcam2022_ptq/crop_index.json \
  --min-conf 0.2 \
  --min-crop-size 32
```

## 2. Export crops and manifests

```bash
python scripts/datasets/iwildcam2022/create_ptq_crop_dataset.py \
  --crop-index /Volumes/Media/iwildcam2022_ptq/crop_index.json \
  --output-root /Volumes/Media/iwildcam2022_ptq \
  --calibration-count 300 \
  --eval-count 50 \
  --seed 0
```

Outputs:

- `crops/calibration/`
- `crops/eval/`
- `manifests/image_manifest.json`
- `manifests/calibration_manifest.json`
- `manifests/eval_manifest.json`
- `manifests/crop_records.json`
- `manifests/selection_summary.json`

The resulting `calibration_manifest.json` and `eval_manifest.json` can be fed directly into the PTQ workflow in `scripts/ptq/`.
