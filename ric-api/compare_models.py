#!/usr/bin/env python3
"""Compare RIC recognition across models × image preprocess pipelines.

Reads images under ../ric_test_data one by one, runs each model under each
preprocess mode, prints results, and writes CSV + JSON summaries.

Preprocess modes:
  baseline     — raw RGB (no preprocess)
  clahe        — CLAHE contrast enhancement only
  canny        — Canny edges on the original grayscale
  clahe_canny  — CLAHE then Canny (same as main.py API)

Ground-truth label comes from the top-level folder under ric_test_data:
  pet1/…       -> pet1
  pp5/…        -> pp5
  other_rics/… -> other  (any pet1/pp5 hit counts as wrong)
  objects/…    -> none   (any detection counts as wrong)

The leaf folder name is the comparison variant (e.g. pet1_print).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA_DIR = REPO_ROOT / "ric_test_data"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

MODELS = {
    "yolov8": PROJECT_ROOT / "yolov8n.pt",
    "ric1-v5": PROJECT_ROOT / "ric1-v5.pt",
    "ric5-v1": PROJECT_ROOT / "ric5-v1.pt",
    "ric1-ric5-v6": PROJECT_ROOT / "ric1-and-ric5-v6.pt",
}

PREPROCESS_ORDER = ("baseline", "clahe", "canny", "clahe_canny")

# Classes that count as RIC predictions for accuracy scoring.
RIC_CLASSES = {"pet1", "pp5"}


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list[int] = field(default_factory=list)


@dataclass
class ImageResult:
    run_id: str
    image: str
    image_id: str
    expected: str
    variant: str
    model: str
    preprocess: str
    conf_threshold: float
    imgsz: int
    predicted: str
    confidence: float
    top1_confidence: float
    top2_confidence: float
    top2_margin: float
    num_detections: int
    score_pet1: float
    score_pp5: float
    score_ric: float
    correct: bool
    detections: list[dict]
    inference_ms: float


def load_raw_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def build_preprocessed(
    path: Path,
    selected: tuple[str, ...] = PREPROCESS_ORDER,
) -> dict[str, np.ndarray]:
    """Build selected preprocess variants from one image read."""
    need_raw = "baseline" in selected
    need_gray = any(name in selected for name in ("clahe", "canny", "clahe_canny"))

    raw = load_raw_rgb(path) if need_raw else None
    gray = load_gray(path) if need_gray else None

    clahe_img = None
    if any(name in selected for name in ("clahe", "clahe_canny")):
        assert gray is not None
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)

    out: dict[str, np.ndarray] = {}
    if "baseline" in selected:
        assert raw is not None
        out["baseline"] = raw
    if "clahe" in selected:
        assert clahe_img is not None
        out["clahe"] = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    if "canny" in selected:
        assert gray is not None
        edges = cv2.Canny(gray, 40, 120)
        out["canny"] = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    if "clahe_canny" in selected:
        assert clahe_img is not None
        edges = cv2.Canny(clahe_img, 40, 120)
        out["clahe_canny"] = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return out


def expected_label_from_path(rel: Path) -> str:
    top = rel.parts[0].lower() if rel.parts else ""
    if top == "pet1":
        return "pet1"
    if top == "pp5":
        return "pp5"
    if top == "other_rics":
        return "other"
    if top == "objects":
        return "none"
    return top or "unknown"


def variant_from_path(rel: Path) -> str:
    """Leaf folder under the top-level label, or the top-level itself."""
    parts = rel.parts
    if len(parts) >= 3:
        return parts[-2]
    if len(parts) == 2:
        return parts[0]
    return "root"


def collect_images(data_dir: Path) -> list[Path]:
    images = [
        p
        for p in sorted(data_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    ]
    return images


def run_predict(
    model: YOLO,
    image: np.ndarray,
    conf: float,
    imgsz: int,
) -> tuple[list[Detection], float]:
    results = model.predict(source=image, imgsz=imgsz, conf=conf, verbose=False)
    result = results[0]
    detections: list[Detection] = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        detections.append(
            Detection(
                label=str(result.names[cls_id]),
                confidence=float(box.conf[0]),
                bbox=[int(x) for x in box.xyxy[0].tolist()],
            )
        )
    detections.sort(key=lambda d: d.confidence, reverse=True)
    inference_ms = float(result.speed.get("inference", 0.0))
    return detections, inference_ms


def pick_prediction(detections: list[Detection], model_name: str) -> tuple[str, float]:
    """Choose the predicted label used for accuracy comparison."""
    if not detections:
        return "none", 0.0

    if model_name == "yolov8":
        # Base COCO model has no RIC classes. Prefer bottle if present; else top hit.
        for det in detections:
            if det.label == "bottle":
                return "bottle", det.confidence
        return detections[0].label, detections[0].confidence

    # Fine-tuned RIC models: top RIC detection wins.
    for det in detections:
        if det.label in RIC_CLASSES:
            return det.label, det.confidence
    return detections[0].label, detections[0].confidence


def is_correct(expected: str, predicted: str, model_name: str) -> bool:
    if model_name == "yolov8":
        # Baseline only: treat bottle as a generic "something detected" signal.
        # It cannot produce pet1/pp5, so score:
        #   pet1/pp5 expected -> correct only if bottle (weak proxy)
        #   other/none expected -> correct if not bottle
        if expected in {"pet1", "pp5"}:
            return predicted == "bottle"
        if expected in {"other", "none"}:
            return predicted != "bottle"
        return False

    if expected in {"pet1", "pp5"}:
        return predicted == expected
    if expected in {"other", "none"}:
        return predicted not in RIC_CLASSES
    return predicted == expected


def derive_detection_features(detections: list[Detection]) -> dict[str, float | int]:
    top1 = float(detections[0].confidence) if detections else 0.0
    top2 = float(detections[1].confidence) if len(detections) > 1 else 0.0
    pet1_score = max(
        (float(d.confidence) for d in detections if d.label == "pet1"),
        default=0.0,
    )
    pp5_score = max(
        (float(d.confidence) for d in detections if d.label == "pp5"),
        default=0.0,
    )
    return {
        "top1_confidence": top1,
        "top2_confidence": top2,
        "top2_margin": top1 - top2,
        "num_detections": len(detections),
        "score_pet1": pet1_score,
        "score_pp5": pp5_score,
        "score_ric": max(pet1_score, pp5_score),
    }


def accuracy_stats(rows: list[ImageResult]) -> dict:
    total = len(rows)
    correct = sum(1 for r in rows if r.correct)
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
    }


def summarize(
    results: list[ImageResult],
    model_names: list[str],
    preprocess_names: list[str],
) -> dict:
    by_model_preprocess: dict[str, dict[str, dict]] = {}
    by_preprocess: dict[str, dict[str, dict]] = {}
    by_variant: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for model_name in model_names:
        by_model_preprocess[model_name] = {}
        for preprocess in preprocess_names:
            rows = [
                r
                for r in results
                if r.model == model_name and r.preprocess == preprocess
            ]
            by_model_preprocess[model_name][preprocess] = accuracy_stats(rows)

    for preprocess in preprocess_names:
        by_preprocess[preprocess] = {}
        for model_name in model_names:
            rows = [
                r
                for r in results
                if r.model == model_name and r.preprocess == preprocess
            ]
            by_preprocess[preprocess][model_name] = accuracy_stats(rows)

    variants = sorted({r.variant for r in results})
    for variant in variants:
        for model_name in model_names:
            for preprocess in preprocess_names:
                rows = [
                    r
                    for r in results
                    if r.variant == variant
                    and r.model == model_name
                    and r.preprocess == preprocess
                ]
                stats = accuracy_stats(rows)
                stats["expected"] = rows[0].expected if rows else ""
                by_variant[variant][model_name][preprocess] = stats

    return {
        "by_model_preprocess": by_model_preprocess,
        "by_preprocess": by_preprocess,
        "by_variant": {
            variant: {model: dict(pre) for model, pre in models.items()}
            for variant, models in by_variant.items()
        },
    }


def print_row(result: ImageResult) -> None:
    mark = "OK" if result.correct else "MISS"
    dets = ", ".join(
        f"{d['label']}:{d['confidence']:.2f}" for d in result.detections[:3]
    ) or "-"
    print(
        f"[{mark}] {result.model:14s} | {result.preprocess:12s} | "
        f"{result.variant:28s} | "
        f"exp={result.expected:6s} pred={result.predicted:10s} "
        f"conf={result.confidence:.2f} | {dets} | {result.image}"
    )


def _fmt_acc(stats: dict | None) -> str:
    if not stats or not stats.get("total"):
        return f"{'n/a':>14s}"
    return f"{stats['accuracy'] * 100:6.1f}% ({stats['correct']}/{stats['total']})"


def print_summary(
    summary: dict,
    model_names: list[str],
    preprocess_names: list[str],
) -> None:
    print("\n" + "=" * 88)
    print("OVERALL ACCURACY BY PREPROCESS × MODEL")
    print("=" * 88)
    header = f"{'preprocess':14s}" + "".join(f" {name:>18s}" for name in model_names)
    print(header)
    print("-" * len(header))
    for preprocess in preprocess_names:
        cells = [
            _fmt_acc(summary["by_preprocess"].get(preprocess, {}).get(name))
            for name in model_names
        ]
        print(f"{preprocess:14s}" + "".join(f" {c:>18s}" for c in cells))

    print("\n" + "=" * 88)
    print("PER-MODEL PREPROCESS COMPARISON")
    print("=" * 88)
    for model_name in model_names:
        print(f"\n  {model_name}")
        for preprocess in preprocess_names:
            stats = summary["by_model_preprocess"].get(model_name, {}).get(preprocess)
            print(f"    {preprocess:12s}  {_fmt_acc(stats).strip()}")

    print("\n" + "=" * 88)
    print("ACCURACY BY VARIANT × MODEL × PREPROCESS")
    print("=" * 88)
    for variant in sorted(summary["by_variant"]):
        models = summary["by_variant"][variant]
        expected = ""
        for model_stats in models.values():
            for stats in model_stats.values():
                if stats.get("expected"):
                    expected = stats["expected"]
                    break
            if expected:
                break
        print(f"\n  {variant}  (expected={expected})")
        sub_header = f"    {'preprocess':12s}" + "".join(
            f" {name:>18s}" for name in model_names
        )
        print(sub_header)
        for preprocess in preprocess_names:
            cells = [
                _fmt_acc(models.get(name, {}).get(preprocess)) for name in model_names
            ]
            print(f"    {preprocess:12s}" + "".join(f" {c:>18s}" for c in cells))


def write_outputs(
    results: list[ImageResult],
    summary: dict,
    out_dir: Path,
    model_paths: dict[str, Path],
    preprocess_names: list[str],
    run_id: str,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"compare_results_{run_id}.csv"
    json_path = out_dir / f"compare_results_{run_id}.json"

    fieldnames = [
        "run_id",
        "image",
        "image_id",
        "expected",
        "variant",
        "model",
        "preprocess",
        "conf_threshold",
        "imgsz",
        "predicted",
        "confidence",
        "top1_confidence",
        "top2_confidence",
        "top2_margin",
        "num_detections",
        "score_pet1",
        "score_pp5",
        "score_ric",
        "correct",
        "detections",
        "inference_ms",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["detections"] = json.dumps(row["detections"], ensure_ascii=False)
            writer.writerow(row)

    payload = {
        "generated_at": run_id,
        "run_id": run_id,
        "models": {k: str(v) for k, v in model_paths.items()},
        "preprocesses": preprocess_names,
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def parse_preprocesses(value: str) -> tuple[str, ...]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("Provide at least one preprocess name")
    unknown = [n for n in names if n not in PREPROCESS_ORDER]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown preprocess(es): {', '.join(unknown)}. "
            f"Choose from: {', '.join(PREPROCESS_ORDER)}"
        )
    # Preserve canonical order even if user passes a subset out of order.
    selected = tuple(name for name in PREPROCESS_ORDER if name in set(names))
    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Test image root (default: {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "compare_output",
        help="Directory for CSV/JSON reports",
    )
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N images (0 = all)",
    )
    p.add_argument(
        "--preprocesses",
        type=parse_preprocesses,
        default=PREPROCESS_ORDER,
        help=(
            "Comma-separated preprocess modes to run. "
            f"Options: {', '.join(PREPROCESS_ORDER)} "
            "(default: all four)"
        ),
    )
    p.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS),
        help=f"Comma-separated model keys (default: {','.join(MODELS)})",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"ERROR: data dir not found: {data_dir}", file=sys.stderr)
        return 1

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown_models = [m for m in selected_models if m not in MODELS]
    if unknown_models:
        print(
            f"ERROR: unknown model(s): {', '.join(unknown_models)}. "
            f"Choose from: {', '.join(MODELS)}",
            file=sys.stderr,
        )
        return 1

    model_paths = {name: MODELS[name] for name in selected_models}
    for name, path in model_paths.items():
        if not path.exists():
            print(f"ERROR: model weights missing for {name}: {path}", file=sys.stderr)
            return 1

    preprocess_names = list(args.preprocesses)

    images = collect_images(data_dir)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(f"ERROR: no images found under {data_dir}", file=sys.stderr)
        return 1

    print(f"Data dir     : {data_dir}")
    print(f"Images       : {len(images)}")
    print(f"Models       : {', '.join(model_paths)}")
    print(f"Preprocesses : {', '.join(preprocess_names)}")
    print(f"Conf         : {args.conf}  imgsz={args.imgsz}")
    print(
        f"Total runs   : {len(images)} images × {len(model_paths)} models × "
        f"{len(preprocess_names)} preprocesses = "
        f"{len(images) * len(model_paths) * len(preprocess_names)}"
    )
    print("-" * 88)

    loaded = {name: YOLO(str(path)) for name, path in model_paths.items()}
    results: list[ImageResult] = []

    for idx, image_path in enumerate(images, start=1):
        rel = image_path.relative_to(data_dir)
        expected = expected_label_from_path(rel)
        variant = variant_from_path(rel)
        print(f"\n({idx}/{len(images)}) {rel}")

        try:
            sources = build_preprocessed(image_path, selected=tuple(preprocess_names))
        except Exception as exc:
            print(f"  SKIP read error: {exc}")
            continue

        for preprocess_name in preprocess_names:
            source = sources[preprocess_name]
            for model_name, model in loaded.items():
                try:
                    detections, inference_ms = run_predict(
                        model, source, conf=args.conf, imgsz=args.imgsz
                    )
                except Exception as exc:
                    print(f"  [{model_name}/{preprocess_name}] ERROR: {exc}")
                    continue

                predicted, confidence = pick_prediction(detections, model_name)
                features = derive_detection_features(detections)
                row = ImageResult(
                    run_id=run_id,
                    image=str(rel),
                    image_id=str(rel),
                    expected=expected,
                    variant=variant,
                    model=model_name,
                    preprocess=preprocess_name,
                    conf_threshold=round(float(args.conf), 4),
                    imgsz=int(args.imgsz),
                    predicted=predicted,
                    confidence=round(confidence, 4),
                    top1_confidence=round(float(features["top1_confidence"]), 4),
                    top2_confidence=round(float(features["top2_confidence"]), 4),
                    top2_margin=round(float(features["top2_margin"]), 4),
                    num_detections=int(features["num_detections"]),
                    score_pet1=round(float(features["score_pet1"]), 4),
                    score_pp5=round(float(features["score_pp5"]), 4),
                    score_ric=round(float(features["score_ric"]), 4),
                    correct=is_correct(expected, predicted, model_name),
                    detections=[asdict(d) for d in detections],
                    inference_ms=round(inference_ms, 2),
                )
                results.append(row)
                print_row(row)

    summary = summarize(results, list(model_paths), preprocess_names)
    print_summary(summary, list(model_paths), preprocess_names)
    csv_path, json_path = write_outputs(
        results,
        summary,
        args.out_dir.resolve(),
        model_paths,
        preprocess_names,
        run_id,
    )
    print(f"\nWrote CSV : {csv_path}")
    print(f"Wrote JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
