#!/usr/bin/env python3
"""
Entry point for the LLM Response Evaluation & Benchmarking System.

Usage:
    python run_evaluation.py [--dataset sample_data/dataset.json] [--outdir outputs] [--llm-judge]

By default this runs fully offline using the built-in HeuristicScorer, so it
works out of the box with no API key. Pass --llm-judge to instead use
Claude as an impartial LLM judge (requires ANTHROPIC_API_KEY to be set and
the `anthropic` package installed).
"""

import argparse
import json
from pathlib import Path

from llm_eval import (
    TestCase,
    ModelResponse,
    EvaluationPipeline,
    HeuristicScorer,
    build_scorer,
    report,
)


def load_dataset(path: str):
    data = json.loads(Path(path).read_text())
    test_cases = [TestCase(**tc) for tc in data["test_cases"]]
    responses = [ModelResponse(**r) for r in data["responses"]]
    return test_cases, responses


def main():
    parser = argparse.ArgumentParser(description="Run the LLM evaluation & benchmarking pipeline.")
    parser.add_argument("--dataset", default="sample_data/dataset.json")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--llm-judge", action="store_true", help="Use Claude as LLM judge instead of heuristics.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.dataset} ...")
    test_cases, responses = load_dataset(args.dataset)
    print(f"  {len(test_cases)} test cases, {len(responses)} responses across "
          f"{len({r.model_name for r in responses})} models.")

    scorer = build_scorer(prefer_llm_judge=args.llm_judge) if args.llm_judge else HeuristicScorer()
    print(f"Using scorer: {scorer.__class__.__name__}")

    pipeline = EvaluationPipeline(scorer=scorer)
    results = pipeline.run(test_cases, responses)

    csv_path = outdir / "raw_scores.csv"
    report.export_csv(results, str(csv_path))
    print(f"Wrote raw per-response scores -> {csv_path}")

    summary_rows = report.per_model_summary(results)
    flagged = report.flagged_responses(results)

    chart_path = outdir / "comparison_chart.png"
    report.plot_comparison_chart(summary_rows, str(chart_path))
    print(f"Wrote comparison chart -> {chart_path}")

    md_path = outdir / "evaluation_report.md"
    report.markdown_report(summary_rows, flagged, str(md_path))
    print(f"Wrote markdown report -> {md_path}")

    print("\n=== Leaderboard ===")
    for i, row in enumerate(pipeline.leaderboard(), start=1):
        print(f"{i}. {row['model_name']:<10} mean overall = {row['mean_overall_score']} (n={row['n']})")

    print(f"\n=== Flagged for human review: {len(flagged)} response(s) ===")
    for f in flagged:
        print(f"  [{f['test_case_id']}] {f['model_name']} -> {f['flags']}")


if __name__ == "__main__":
    main()
