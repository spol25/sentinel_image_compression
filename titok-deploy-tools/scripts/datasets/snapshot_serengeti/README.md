# Snapshot Serengeti PTQ Crop Prep

This folder contains the source-specific pipeline for building a PTQ-ready crop dataset from Snapshot Serengeti.

Why this dataset:

- images are exposed through cloud storage prefixes, so selective download is practical
- the dataset publishes bounding boxes separately
- metadata includes fields like `location`, `seq_id`, `datetime`, and often species labels

The workflow is:

1. download the metadata JSON and bounding-box JSON
2. build a crop candidate index from the bounding boxes
3. select a representative non-overlapping `300` calibration + `50` eval crop set
4. selectively download only those source images
5. crop them locally and write `image_manifest.json`, `calibration_manifest.json`, and `eval_manifest.json`

## 1. Build the candidate index

```bash
python scripts/datasets/snapshot_serengeti/build_bbox_index.py \
  --metadata-json /path/to/snapshot_serengeti_metadata.json \
  --bboxes-json /path/to/snapshot_serengeti_bboxes.json \
  --output /Volumes/Media/snapshot_serengeti_ptq/candidate_index.json
```

## 2. Selectively download and crop the PTQ dataset

```bash
python scripts/datasets/snapshot_serengeti/create_ptq_crop_dataset.py \
  --candidate-index /Volumes/Media/snapshot_serengeti_ptq/candidate_index.json \
  --output-root /Volumes/Media/snapshot_serengeti_ptq \
  --calibration-count 300 \
  --eval-count 50 \
  --seed 0
```

Outputs:

- `downloads/` with only the selected source images
- `crops/calibration/`
- `crops/eval/`
- `manifests/image_manifest.json`
- `manifests/calibration_manifest.json`
- `manifests/eval_manifest.json`
- `manifests/crop_records.json`
- `manifests/selection_summary.json`

The selection tries to be representative by diversifying across:

- species/category where available
- locations
- seasons
- sequence IDs to reduce near-duplicates
