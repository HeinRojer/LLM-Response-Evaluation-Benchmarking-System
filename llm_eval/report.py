"""
Reporting: turns a list of EvaluationResult objects into
  - a CSV of raw per-response scores
  - a per-model summary table (mean score per dimension + overall)
  - a bar chart comparing models across dimensions
  - a markdown summary report
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import List

from .data_models import DIMENSIONS, EvaluationResult


def export_csv(results: List[EvaluationResult], path: str) -> None:
    path = Path(path)
    fieldnames = ["test_case_id", "model_name", "overall_score", *DIMENSIONS, "flags"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_row())


def per_model_summary(results: List[EvaluationResult]) -> List[dict]:
    """Mean score per dimension + overall, grouped by model."""
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        for dim, ds in r.scores.items():
            agg[r.model_name][dim].append(ds.score)
        agg[r.model_name]["overall"].append(r.overall_score)

    rows = []
    for model, dims in agg.items():
        row = {"model_name": model}
        for dim in DIMENSIONS:
            vals = dims[dim]
            row[dim] = round(sum(vals) / len(vals), 2) if vals else None
        row["overall"] = round(sum(dims["overall"]) / len(dims["overall"]), 2)
        row["n_responses"] = len(dims["overall"])
        rows.append(row)
    return sorted(rows, key=lambda r: r["overall"], reverse=True)


def flagged_responses(results: List[EvaluationResult]) -> List[dict]:
    """All responses that were auto-flagged for human follow-up review."""
    return [r.to_row() for r in results if r.flags]


def plot_comparison_chart(summary_rows: List[dict], out_path: str) -> str:
    """Grouped bar chart: one group per dimension, one bar per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = [r["model_name"] for r in summary_rows]
    dims = DIMENSIONS
    x = np.arange(len(dims))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, row in enumerate(summary_rows):
        values = [row[d] for d in dims]
        ax.bar(x + i * width, values, width, label=row["model_name"])

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=9)
    ax.set_ylabel("Mean score (1-5, higher = better)")
    ax.set_ylim(0, 5)
    ax.set_title("LLM Response Evaluation — Dimension Comparison")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def markdown_report(summary_rows: List[dict], flagged: List[dict], out_path: str) -> str:
    lines = ["# LLM Evaluation & Benchmarking Report", ""]
    lines.append("## Leaderboard (mean overall score, 1-5)")
    lines.append("")
    lines.append("| Rank | Model | Overall | " + " | ".join(DIMENSIONS) + " | N |")
    lines.append("|---" * (4 + len(DIMENSIONS)) + "|")
    for i, row in enumerate(summary_rows, start=1):
        dim_vals = " | ".join(str(row[d]) for d in DIMENSIONS)
        lines.append(f"| {i} | {row['model_name']} | {row['overall']} | {dim_vals} | {row['n_responses']} |")

    lines.append("")
    lines.append(f"## Flagged Responses ({len(flagged)})")
    lines.append("")
    if flagged:
        lines.append("| Test Case | Model | Overall | Flags |")
        lines.append("|---|---|---|---|")
        for f in flagged:
            lines.append(f"| {f['test_case_id']} | {f['model_name']} | {f['overall_score']} | {f['flags']} |")
    else:
        lines.append("_No responses were auto-flagged for review._")

    Path(out_path).write_text("\n".join(lines))
    return out_path
