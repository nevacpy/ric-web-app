#!/usr/bin/env python3
"""Generate Phase 1 analytics charts from compare_models outputs.

Reads a compare CSV (or auto-picks latest from compare_output) and writes:
- PNG charts (accuracy, error analysis, latency)
- index.html gallery
- run_summary.md and summary.json

Usage:
  python generate_analytics.py
  python generate_analytics.py --input-csv ./compare_output/compare_results_20260801T010101Z.csv
  python generate_analytics.py --input-dir ./compare_output --out-dir ./analytics_output
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
except Exception as exc:  # pragma: no cover - import guard for runtime ergonomics
    print(
        "ERROR: Missing analytics dependencies. Install with:\n"
        "  pip install pandas matplotlib seaborn numpy\n"
        f"Import failure: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1)


EXPECTED_BASELINE = ["pet1", "pp5", "other", "none"]
METRICS_CLASSES = ["pet1", "pp5"]


@dataclass
class Paths:
    csv_path: Path
    out_dir: Path
    run_id: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Path to compare_results_*.csv (default: latest under --input-dir)",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "compare_output",
        help="Directory containing compare_results_*.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "analytics_output",
        help="Directory for generated analytics artifacts",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional run id label used in output folder naming",
    )
    return p.parse_args()


def _extract_run_id(path: Path) -> str:
    m = re.search(r"compare_results_(\d{8}T\d{6}Z)", path.name)
    if m:
        return m.group(1)
    return path.stem


def resolve_paths(args: argparse.Namespace) -> Paths:
    if args.input_csv:
        csv_path = args.input_csv.resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    else:
        input_dir = args.input_dir.resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        candidates = sorted(input_dir.glob("compare_results_*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No compare_results_*.csv found under {input_dir}. Run compare_models.py first."
            )
        csv_path = candidates[-1]

    run_id = args.run_id.strip() or _extract_run_id(csv_path)
    out_dir = (args.out_dir.resolve() / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return Paths(csv_path=csv_path, out_dir=out_dir, run_id=run_id)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "image",
        "expected",
        "variant",
        "model",
        "preprocess",
        "predicted",
        "confidence",
        "correct",
        "inference_ms",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

    df["expected"] = df["expected"].fillna("unknown").astype(str)
    df["predicted"] = df["predicted"].fillna("unknown").astype(str)
    df["variant"] = df["variant"].fillna("unknown").astype(str)
    df["model"] = df["model"].fillna("unknown").astype(str)
    df["preprocess"] = df["preprocess"].fillna("unknown").astype(str)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["inference_ms"] = pd.to_numeric(df["inference_ms"], errors="coerce").fillna(0.0)

    # Accept bools, "True"/"False", 1/0 gracefully.
    if df["correct"].dtype == bool:
        pass
    else:
        df["correct"] = (
            df["correct"].astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
        )

    return df


def combo_order(df: pd.DataFrame) -> list[str]:
    combos = (
        df[["model", "preprocess"]]
        .drop_duplicates()
        .sort_values(["model", "preprocess"])
        .apply(lambda row: f"{row.model} | {row.preprocess}", axis=1)
        .tolist()
    )
    return combos


def ensure_combo_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["combo"] = out["model"] + " | " + out["preprocess"]
    return out


def labels_for_confusion(df: pd.DataFrame) -> list[str]:
    labels = sorted(set(df["expected"]).union(set(df["predicted"])))
    # Keep familiar class ordering first if present.
    ordered = [name for name in EXPECTED_BASELINE if name in labels]
    tail = [name for name in labels if name not in ordered]
    return ordered + tail


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def chart_confusion_by_model_preprocess(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    saved: list[Path] = []
    labels = labels_for_confusion(df)

    for model in sorted(df["model"].unique()):
        model_df = df[df["model"] == model]
        preprocesses = sorted(model_df["preprocess"].unique())
        n = len(preprocesses)
        fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 6), 4.8), squeeze=False)

        for i, pre in enumerate(preprocesses):
            ax = axes[0, i]
            sub = model_df[model_df["preprocess"] == pre]
            cm = pd.crosstab(sub["expected"], sub["predicted"], dropna=False)
            cm = cm.reindex(index=labels, columns=labels, fill_value=0)

            # Normalize per expected class for readability.
            cm_norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            sns.heatmap(
                cm_norm,
                annot=cm,
                fmt="d",
                cmap="YlGnBu",
                cbar=i == n - 1,
                vmin=0,
                vmax=1,
                ax=ax,
            )
            ax.set_title(f"{model}\n{pre}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Expected")

        fig.suptitle("Confusion Matrix (annot=count, color=normalized)", y=1.03)
        out = out_dir / f"confusion_{_safe_name(model)}.png"
        save_figure(fig, out)
        saved.append(out)

    return saved


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def metrics_by_combo(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for combo, sub in df.groupby("combo", sort=False):
        for klass in METRICS_CLASSES:
            tp = int(((sub["expected"] == klass) & (sub["predicted"] == klass)).sum())
            fp = int(((sub["expected"] != klass) & (sub["predicted"] == klass)).sum())
            fn = int(((sub["expected"] == klass) & (sub["predicted"] != klass)).sum())
            p, r, f1 = _prf1(tp, fp, fn)
            rows.append(
                {
                    "combo": combo,
                    "class": klass,
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "support": int((sub["expected"] == klass).sum()),
                }
            )
    return pd.DataFrame(rows)


def chart_prf1(df: pd.DataFrame, out_dir: Path) -> Path:
    metrics = metrics_by_combo(df)
    melted = metrics.melt(
        id_vars=["combo", "class", "support"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="value",
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    class_order = METRICS_CLASSES
    for i, klass in enumerate(class_order):
        ax = axes[i]
        sub = melted[melted["class"] == klass]
        sns.barplot(data=sub, x="combo", y="value", hue="metric", ax=ax, palette="Set2")
        ax.set_ylim(0, 1.0)
        support = int(metrics[metrics["class"] == klass]["support"].max() or 0)
        ax.set_title(f"{klass} metrics (support={support})")
        ax.set_ylabel("Score")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        if i == 0:
            ax.legend(loc="upper right")
        else:
            ax.get_legend().remove()

    fig.suptitle("Precision / Recall / F1 by Model + Preprocess", y=1.02)
    out = out_dir / "metrics_prf1_by_combo.png"
    save_figure(fig, out)
    return out


def chart_accuracy_by_variant(df: pd.DataFrame, out_dir: Path) -> Path:
    piv = (
        df.groupby(["variant", "combo"], dropna=False)["correct"].mean().reset_index()
        .pivot(index="variant", columns="combo", values="correct")
        .fillna(np.nan)
    )

    fig_h = max(5, 0.5 * len(piv.index) + 2)
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(piv.columns) + 4), fig_h))
    sns.heatmap(
        piv,
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
        vmin=0,
        vmax=1,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )
    ax.set_title("Accuracy by Variant x Model+Preprocess")
    ax.set_xlabel("Model + Preprocess")
    ax.set_ylabel("Variant")

    out = out_dir / "accuracy_by_variant_heatmap.png"
    save_figure(fig, out)
    return out


def chart_confidence_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="correct",
        y="confidence",
        hue="correct",
        legend=False,
        ax=ax,
        palette="Set3",
    )
    ax.set_title("Confidence Distribution: Correct vs Incorrect")
    ax.set_xlabel("Correct")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, max(1.0, float(df["confidence"].max()) + 0.02))

    out = out_dir / "confidence_correct_vs_incorrect_boxplot.png"
    save_figure(fig, out)
    return out


def chart_latency_boxplot(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="model",
        y="inference_ms",
        hue="model",
        legend=False,
        ax=ax,
        palette="crest",
    )
    ax.set_title("Inference Latency by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Inference (ms)")
    ax.tick_params(axis="x", rotation=15)

    out = out_dir / "latency_by_model_boxplot.png"
    save_figure(fig, out)
    return out


def chart_accuracy_vs_latency(df: pd.DataFrame, out_dir: Path) -> Path:
    agg = (
        df.groupby("combo", dropna=False)
        .agg(accuracy=("correct", "mean"), latency_ms=("inference_ms", "median"), n=("image", "count"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=agg, x="latency_ms", y="accuracy", size="n", legend=False, s=140, ax=ax)

    for _, row in agg.iterrows():
        ax.annotate(row["combo"], (row["latency_ms"], row["accuracy"]), fontsize=8, alpha=0.85)

    ax.set_title("Accuracy vs Median Latency (Pareto view)")
    ax.set_xlabel("Median Inference (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)

    out = out_dir / "accuracy_vs_latency_scatter.png"
    save_figure(fig, out)
    return out


def chart_hardest_variants(df: pd.DataFrame, out_dir: Path, top_n: int = 12) -> Path:
    agg = (
        df.groupby("variant", dropna=False)
        .agg(accuracy=("correct", "mean"), n=("image", "count"))
        .reset_index()
    )
    agg = agg[agg["n"] > 0].sort_values(["accuracy", "n"], ascending=[True, False]).head(top_n)

    fig_h = max(4, 0.45 * len(agg) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    sns.barplot(
        data=agg,
        y="variant",
        x="accuracy",
        hue="variant",
        legend=False,
        palette="rocket",
        ax=ax,
    )
    ax.set_title(f"Hardest Variants (lowest accuracy, top {len(agg)})")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Variant")
    ax.set_xlim(0, 1.0)

    out = out_dir / "hardest_variants_bar.png"
    save_figure(fig, out)
    return out


def chart_scatter_confidence_vs_latency(df: pd.DataFrame, out_dir: Path) -> Path:
    # Reduce visual noise while preserving useful trends.
    sample = df.copy()
    if len(sample) > 6000:
        sample = sample.sample(n=6000, random_state=42)

    sample["correct_label"] = np.where(sample["correct"], "correct", "incorrect")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.scatterplot(
        data=sample,
        x="inference_ms",
        y="confidence",
        hue="correct_label",
        style="model",
        alpha=0.55,
        s=42,
        ax=ax,
        palette={"correct": "#1f7a4d", "incorrect": "#c44e52"},
    )

    ax.set_title("Per-image Confidence vs Inference Time")
    ax.set_xlabel("Inference (ms)")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, max(1.0, float(sample["confidence"].max()) + 0.03))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    out = out_dir / "scatter_confidence_vs_latency.png"
    save_figure(fig, out)
    return out


def chart_stacked_accuracy_by_combo(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = (
        df.groupby(["combo", "correct"], dropna=False)
        .size()
        .reset_index(name="count")
        .pivot(index="combo", columns="correct", values="count")
        .fillna(0)
    )

    for col in [False, True]:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts.reindex(columns=[False, True], fill_value=0)
    totals = counts.sum(axis=1).replace(0, np.nan)
    pct = counts.div(totals, axis=0).fillna(0.0)

    fig_h = max(5, 0.45 * len(pct) + 2)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    ax.barh(
        pct.index,
        pct[False],
        color="#d96c75",
        label="Incorrect",
    )
    ax.barh(
        pct.index,
        pct[True],
        left=pct[False],
        color="#3a9d6d",
        label="Correct",
    )

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Share of predictions")
    ax.set_ylabel("Model + Preprocess")
    ax.set_title("100% Stacked Accuracy Composition by Model+Preprocess")
    ax.legend(loc="lower right")

    out = out_dir / "stacked_accuracy_share_by_combo.png"
    save_figure(fig, out)
    return out


def chart_stacked_prediction_mix(df: pd.DataFrame, out_dir: Path, top_k: int = 6) -> Path:
    pred_freq = df["predicted"].value_counts()
    top_labels = pred_freq.head(top_k).index.tolist()

    mix = df.copy()
    mix["predicted_group"] = np.where(
        mix["predicted"].isin(top_labels),
        mix["predicted"],
        "other",
    )

    piv = (
        mix.groupby(["combo", "predicted_group"], dropna=False)
        .size()
        .reset_index(name="count")
        .pivot(index="combo", columns="predicted_group", values="count")
        .fillna(0)
    )

    # Convert to shares for better cross-combo comparison.
    piv = piv.div(piv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    piv = piv.reindex(sorted(piv.index))

    fig_h = max(5, 0.5 * len(piv) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    left = np.zeros(len(piv))
    colors = sns.color_palette("Spectral", n_colors=max(3, len(piv.columns)))
    for idx, col in enumerate(piv.columns):
        vals = piv[col].to_numpy()
        ax.barh(piv.index, vals, left=left, label=str(col), color=colors[idx], alpha=0.95)
        left += vals

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Predicted label share")
    ax.set_ylabel("Model + Preprocess")
    ax.set_title("100% Stacked Predicted Label Mix by Model+Preprocess")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="Predicted")

    out = out_dir / "stacked_prediction_mix_by_combo.png"
    save_figure(fig, out)
    return out


def build_summary(df: pd.DataFrame, run_id: str) -> dict:
    combo = (
        df.groupby("combo", dropna=False)
        .agg(
            total=("image", "count"),
            accuracy=("correct", "mean"),
            median_latency_ms=("inference_ms", "median"),
            p95_latency_ms=("inference_ms", lambda s: float(np.percentile(s, 95))),
            confidence_median=("confidence", "median"),
        )
        .sort_values(["accuracy", "median_latency_ms"], ascending=[False, True])
        .reset_index()
    )

    hardest = (
        df.groupby("variant", dropna=False)
        .agg(total=("image", "count"), accuracy=("correct", "mean"))
        .sort_values(["accuracy", "total"], ascending=[True, False])
        .head(10)
        .reset_index()
    )

    summary = {
        "run_id": run_id,
        "rows": int(len(df)),
        "images": int(df["image"].nunique()),
        "models": sorted(df["model"].unique().tolist()),
        "preprocesses": sorted(df["preprocess"].unique().tolist()),
        "overall_accuracy": float(df["correct"].mean()),
        "combo_metrics": combo.to_dict(orient="records"),
        "hardest_variants": hardest.to_dict(orient="records"),
    }
    return summary


def write_markdown_summary(summary: dict, out_dir: Path, csv_path: Path) -> Path:
    md_path = out_dir / "run_summary.md"
    lines: list[str] = []
    lines.append(f"# Analytics Summary: {summary['run_id']}")
    lines.append("")
    lines.append(f"Source CSV: `{csv_path}`")
    lines.append("")
    lines.append(f"- Total rows: {summary['rows']}")
    lines.append(f"- Unique images: {summary['images']}")
    lines.append(f"- Overall accuracy: {summary['overall_accuracy']:.3f}")
    lines.append("")

    lines.append("## Best model+preprocess combinations")
    lines.append("")
    lines.append("| combo | accuracy | median_latency_ms | p95_latency_ms | n |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in summary["combo_metrics"][:10]:
        lines.append(
            "| "
            f"{row['combo']} | {row['accuracy']:.3f} | {row['median_latency_ms']:.2f} | "
            f"{row['p95_latency_ms']:.2f} | {row['total']} |"
        )

    lines.append("")
    lines.append("## Hardest variants")
    lines.append("")
    lines.append("| variant | accuracy | n |")
    lines.append("|---|---:|---:|")
    for row in summary["hardest_variants"]:
        lines.append(f"| {row['variant']} | {row['accuracy']:.3f} | {row['total']} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def write_html_gallery(out_dir: Path, run_id: str, image_paths: Iterable[Path]) -> Path:
    rels = [p.name for p in image_paths]
    cards = "\n".join(
        f"<article class='card'><h2>{name}</h2><img src='{name}' alt='{name}' /></article>"
        for name in rels
    )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>RIC Analytics {run_id}</title>
  <style>
    :root {{
      --bg: #f6f9f4;
      --ink: #17351f;
      --card: #ffffff;
      --line: #d5e3d2;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, Segoe UI, sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at 20% 0%, #e4f0dd 0%, var(--bg) 45%);
    }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 1.8rem; }}
    p {{ margin: 0 0 18px; opacity: 0.85; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid var(--line); border-radius: 12px; background: var(--card); padding: 12px; }}
    .card h2 {{ font-size: 0.95rem; margin: 0 0 10px; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 8px; }}
  </style>
</head>
<body>
  <main>
    <h1>RIC Analytics Chart Pack</h1>
    <p>Run: {run_id}</p>
    <section class=\"grid\">{cards}</section>
  </main>
</body>
</html>
"""
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def main() -> int:
    args = parse_args()
    sns.set_theme(style="whitegrid")

    try:
        paths = resolve_paths(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Using CSV : {paths.csv_path}")
    print(f"Run id    : {paths.run_id}")
    print(f"Output dir: {paths.out_dir}")

    try:
        df = load_data(paths.csv_path)
    except Exception as exc:
        print(f"ERROR reading CSV: {exc}", file=sys.stderr)
        return 1

    df = ensure_combo_col(df)

    outputs: list[Path] = []
    outputs.extend(chart_confusion_by_model_preprocess(df, paths.out_dir))
    outputs.append(chart_prf1(df, paths.out_dir))
    outputs.append(chart_accuracy_by_variant(df, paths.out_dir))
    outputs.append(chart_confidence_distribution(df, paths.out_dir))
    outputs.append(chart_latency_boxplot(df, paths.out_dir))
    outputs.append(chart_accuracy_vs_latency(df, paths.out_dir))
    outputs.append(chart_hardest_variants(df, paths.out_dir))
    outputs.append(chart_scatter_confidence_vs_latency(df, paths.out_dir))
    outputs.append(chart_stacked_accuracy_by_combo(df, paths.out_dir))
    outputs.append(chart_stacked_prediction_mix(df, paths.out_dir))

    summary = build_summary(df, paths.run_id)
    summary_path = paths.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path = write_markdown_summary(summary, paths.out_dir, paths.csv_path)
    html_path = write_html_gallery(paths.out_dir, paths.run_id, outputs)

    print("\nGenerated charts:")
    for path in outputs:
        print(f"  - {path}")
    print(f"Summary JSON : {summary_path}")
    print(f"Summary MD   : {md_path}")
    print(f"HTML Gallery : {html_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
