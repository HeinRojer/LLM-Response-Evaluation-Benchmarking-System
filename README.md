# LLM-Response-Evaluation-Benchmarking-System


# LLM Response Evaluation & Benchmarking System

A Python-based evaluation workflow for benchmarking LLM responses across
**factual accuracy, relevance, reasoning quality, hallucination risk, and
instruction adherence** — enabling structured, consistent comparison of
multiple models on the same prompt set.

## What it does

- **Multi-dimensional scoring framework** — every response is graded against
  the same five-dimension rubric (1-5 scale, weighted), so scores are
  standardized and comparable across models and reviewers instead of being
  ad-hoc or reviewer-dependent.
- **Automated evaluation & comparative analysis** — a pipeline scores every
  (prompt, model, response) triple, ranks models on a leaderboard, and
  auto-flags responses likely to have hallucinations, weak reasoning, or
  missed instructions — cutting manual review down to just the flagged
  subset instead of every response.
- **Two interchangeable scorers**, behind one interface:
  - `LLMJudgeScorer` — uses Claude as an impartial judge with a standardized
    grading prompt (recommended; needs `ANTHROPIC_API_KEY`).
  - `HeuristicScorer` — a dependency-light, offline fallback (lexical
    overlap, instruction-pattern parsing, fabrication-marker detection) so
    the whole pipeline runs with zero API cost for demos, CI, or dev work.
- **Reporting** — CSV of raw scores, a per-model summary table, a grouped
  bar chart comparing models dimension-by-dimension, and a Markdown report
  listing every flagged response for human follow-up.

## Project layout

```
llm-eval-benchmark/
├── llm_eval/
│   ├── data_models.py   # TestCase, ModelResponse, DimensionScore, EvaluationResult
│   ├── scorer.py        # BaseScorer, LLMJudgeScorer, HeuristicScorer
│   ├── evaluator.py      # EvaluationPipeline: orchestration + auto-flagging
│   └── report.py         # CSV export, summary table, chart, markdown report
├── sample_data/
│   └── dataset.json      # 5 prompts x 3 simulated models = 15 responses
├── tests/
│   └── test_pipeline.py  # pytest suite (6 tests)
├── run_evaluation.py      # CLI entry point
└── outputs/               # generated: raw_scores.csv, comparison_chart.png, evaluation_report.md
```

## Quickstart

```bash
pip install -r requirements.txt

# Offline, no API key needed (uses HeuristicScorer):
python run_evaluation.py

# Use Claude as an LLM judge instead (needs ANTHROPIC_API_KEY):
export ANTHROPIC_API_KEY=sk-...
python run_evaluation.py --llm-judge

# Run the test suite
pytest tests/ -v
```

Outputs land in `outputs/`:
- `raw_scores.csv` — every response's per-dimension + overall score
- `comparison_chart.png` — bar chart comparing models across all 5 dimensions
- `evaluation_report.md` — leaderboard + table of flagged responses

## Using it on your own data

Swap in your own prompts and model outputs by building `TestCase` and
`ModelResponse` objects (or editing `sample_data/dataset.json`, which the
CLI loads directly):

```python
from llm_eval import TestCase, ModelResponse, EvaluationPipeline, build_scorer, report

test_cases = [
    TestCase(id="q1", prompt="Explain quantum entanglement simply.",
              reference_answer="...", instructions=["Under 3 sentences"]),
]
responses = [
    ModelResponse(test_case_id="q1", model_name="gpt-4o", response_text="..."),
    ModelResponse(test_case_id="q1", model_name="claude", response_text="..."),
]

pipeline = EvaluationPipeline(scorer=build_scorer(prefer_llm_judge=True))
results = pipeline.run(test_cases, responses)

print(pipeline.leaderboard())
report.export_csv(results, "outputs/raw_scores.csv")
```

## Design notes

- **Why a weighted 1-5 rubric instead of a single pass/fail score?**
  A single score collapses failure modes together — a fluent-but-wrong
  answer and a hedgy-but-accurate one would look identical. Separate
  dimensions make it possible to see *why* a model underperforms, not just
  that it does, and `DIMENSION_WEIGHTS` in `data_models.py` can be retuned
  per use case (e.g. weight `hallucination_risk` higher for a
  factual-QA benchmark).
- **Why normalize hallucination_risk so 5 = good?** Keeping every dimension
  "higher is better" lets the weighted average and the charting code treat
  all five dimensions identically, with no special-casing.
- **Why auto-flagging instead of scoring everything by hand?** The scoring
  pipeline is built to reduce manual assessment effort: `evaluator.py`
  applies threshold rules (`FLAG_THRESHOLDS`) so only responses that look
  genuinely risky surface for human review, rather than requiring a human
  to read all N x M responses.
