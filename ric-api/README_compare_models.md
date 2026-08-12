# Model comparison (`compare_models.py`)

Compares RIC recognition accuracy across three YOLO models on images in `ric_test_data/`.

| Model key       | Weights file              | Classes                          |
|-----------------|---------------------------|----------------------------------|
| `yolov8`        | `yolov8n.pt`              | COCO (baseline; no RIC classes)  |
| `ric1-v5`       | `ric1-v5.pt`              | `pet1`                           |
| `ric5-v1`       | `ric5-v1.pt`              | `pp5`                            |
| `ric1-ric5-v6`  | `ric1-and-ric5-v6.pt`     | `pet1`, `pp5`                    |

## Prerequisites

1. Activate the project virtualenv (from `rnapp-api/`):

```bash
cd rnapp-api
source .venv/bin/activate
```

2. Ensure dependencies are installed (`ultralytics`, `opencv-python`, `numpy`).

3. Place weight files in `rnapp-api/`:

- `yolov8n.pt` (downloaded automatically by Ultralytics on first use if missing)
- `ric1-v5.pt`
- `ric5-v1.pt`
- `ric1-and-ric5-v6.pt`

4. Keep test images under `../ric_test_data` (repo root).

## Quick start

```bash
cd rnapp-api
source .venv/bin/activate
python compare_models.py
```

Smoke test on a few images:

```bash
python compare_models.py --limit 5
```

## How labels and variants are read

Folder layout under `ric_test_data/` is the ground truth:

```text
ric_test_data/
  pet1/
    pet1_print/                 ← variant name
    pet1_transparent_embossed/
    ...
  pp5/
    pp5_printed/
    pp5_transluent_embossed/
    ...
  other_rics/                   ← variant = other_rics
  objects/
```

| Top-level folder | Expected label | Correct when…                                      |
|------------------|----------------|----------------------------------------------------|
| `pet1/`          | `pet1`         | Top RIC prediction is `pet1`                       |
| `pp5/`           | `pp5`          | Top RIC prediction is `pp5`                        |
| `other_rics/`    | `other`        | No `pet1` / `pp5` detection                        |
| `objects/`       | `none`         | No `pet1` / `pp5` detection                        |

The **leaf folder name** (e.g. `pet1_print`) is the **variant** used in the per-folder accuracy table.

### Baseline YOLOv8 scoring

Base YOLOv8 has no RIC classes. The script uses a weak proxy:

- For `pet1` / `pp5` images: correct if it detects `bottle`
- For `other` / `none`: correct if it does **not** detect `bottle`

Fine-tuned models are the meaningful accuracy comparison; YOLOv8 is a baseline only.

## Preprocessing

- **Fine-tuned models** always use the same CLAHE + Canny edge pipeline as `main.py`.
- **Base YOLOv8** uses raw RGB by default (COCO training distribution).
- Pass `--yolo-preprocess` to feed edge maps to base YOLOv8 as well.

## Options

| Flag                | Default                         | Description                                      |
|---------------------|---------------------------------|--------------------------------------------------|
| `--data-dir PATH`   | `../ric_test_data`              | Root folder of test images                       |
| `--out-dir PATH`    | `./compare_output`              | Where CSV/JSON reports are written               |
| `--conf FLOAT`      | `0.25`                          | Confidence threshold                             |
| `--imgsz INT`       | `640`                           | Inference image size                             |
| `--limit N`         | `0` (all)                       | Process only the first N images                  |
| `--yolo-preprocess` | off                             | Run base YOLOv8 on CLAHE+Canny edges             |

Examples:

```bash
# Full run, custom confidence
python compare_models.py --conf 0.35

# Custom data / output paths
python compare_models.py --data-dir ../ric_test_data --out-dir ./compare_output

# Edge-preprocess the baseline too
python compare_models.py --yolo-preprocess
```

## Output

### Console

For each image, one line per model:

```text
[OK]   ric1-ric5-v6 | pet1_print | exp=pet1 pred=pet1 conf=0.87 | pet1:0.87 | pet1/pet1_print/IMG_6308.jpg
[MISS] ric1-v5      | pp5_printed | exp=pp5  pred=none conf=0.00 | - | pp5/pp5_printed/IMG_6338.jpg
```

After all images:

1. **Overall accuracy** per model (`correct/total`)
2. **Accuracy by variant** (folder name) across models

### Files

Written under `compare_output/` (timestamped):

| File | Contents |
|------|----------|
| `compare_results_<UTC>.csv` | One row per image × model |
| `compare_results_<UTC>.json` | Full results + summary aggregates |

CSV columns: `run_id`, `image`, `image_id`, `expected`, `variant`, `model`, `preprocess`, `conf_threshold`, `imgsz`, `predicted`, `confidence`, `top1_confidence`, `top2_confidence`, `top2_margin`, `num_detections`, `score_pet1`, `score_pp5`, `score_ric`, `correct`, `detections`, `inference_ms`.

New advanced-analysis fields:

- `score_pet1`, `score_pp5`: best confidence seen for each class per image/model/preprocess row.
- `score_ric`: `max(score_pet1, score_pp5)` for one-vs-rest style analysis.
- `top2_margin`: confidence gap between top-1 and top-2 detections.
- `num_detections`: total detections emitted for the row.
- `run_id`, `conf_threshold`, `imgsz`: run metadata for reproducible experiments.

## Phase 1 analytics charts

Generate a chart pack from the latest comparison CSV:

```bash
cd rnapp-api
source .venv/bin/activate
python generate_analytics.py
```

From the project root, you can also run:

```bash
pnpm run analytics
```

Open the latest generated gallery:

```bash
pnpm run analytics:open
```

Run analytics, then open gallery automatically:

```bash
pnpm run analytics:run-open
```

Use a specific run:

```bash
python generate_analytics.py \
  --input-csv ./compare_output/compare_results_20260722T020628Z.csv
```

Or from the root:

```bash
pnpm run analytics:csv -- ./ric-api/compare_output/compare_results_20260722T020628Z.csv
```

Custom output path:

```bash
python generate_analytics.py --out-dir ./analytics_output
```

Artifacts are written to `analytics_output/<run_id>/`:

| File | Contents |
|------|----------|
| `confusion_<model>.png` | Confusion heatmaps split by preprocess for each model |
| `metrics_prf1_by_combo.png` | Precision / recall / F1 for `pet1` and `pp5` |
| `accuracy_by_variant_heatmap.png` | Variant-level accuracy map |
| `confidence_correct_vs_incorrect_boxplot.png` | Confidence separation for right vs wrong predictions |
| `latency_by_model_boxplot.png` | Inference-time distribution by model |
| `accuracy_vs_latency_scatter.png` | Accuracy/latency tradeoff per model+preprocess |
| `hardest_variants_bar.png` | Lowest-accuracy variants |
| `scatter_confidence_vs_latency.png` | Per-image confidence vs inference time (colored by correctness) |
| `stacked_accuracy_share_by_combo.png` | 100% stacked correct vs incorrect share by model+preprocess |
| `stacked_prediction_mix_by_combo.png` | 100% stacked predicted-label composition by model+preprocess |
| `summary.json` | Machine-readable run summary |
| `run_summary.md` | Human-readable summary tables |
| `index.html` | Browser gallery of generated charts |

If dependencies are missing, install:

```bash
pip install pandas matplotlib seaborn numpy
```

## Tips

- A full pass over ~190 images × 4 models can take several minutes on CPU.
- Use `--limit` while iterating on thresholds or preprocessing.
- Re-run with different `--conf` values if you want a sensitivity check; each run creates a new timestamped report.
