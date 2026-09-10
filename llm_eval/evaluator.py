"""
Automated evaluation & comparative-analysis pipeline.

Takes a set of TestCases plus one or more ModelResponses per test case,
runs them through a BaseScorer, flags likely issues (hallucination risk,
weak instruction-following, low reasoning quality), and produces a flat
list of EvaluationResult objects ready for reporting.
"""

from typing import Dict, Iterable, List

from .data_models import EvaluationResult, ModelResponse, TestCase
from .scorer import BaseScorer, build_scorer

# Thresholds used to auto-flag responses for human follow-up review.
FLAG_THRESHOLDS = {
    "hallucination_risk": 3,      # score <= 3 -> flag possible hallucination
    "instruction_adherence": 2,   # score <= 2 -> flag instruction miss
    "reasoning_quality": 2,       # score <= 2 -> flag weak reasoning
}


class EvaluationPipeline:
    def __init__(self, scorer: BaseScorer = None):
        self.scorer = scorer or build_scorer(prefer_llm_judge=True)
        self.results: List[EvaluationResult] = []

    def run(
        self,
        test_cases: Iterable[TestCase],
        responses: Iterable[ModelResponse],
    ) -> List[EvaluationResult]:
        """Score every response against its matching test case."""
        tc_by_id: Dict[str, TestCase] = {tc.id: tc for tc in test_cases}
        self.results = []

        for resp in responses:
            tc = tc_by_id.get(resp.test_case_id)
            if tc is None:
                raise ValueError(f"No test case found for id={resp.test_case_id!r}")

            dim_scores = self.scorer.score(tc, resp)
            overall = self.scorer.weighted_overall(dim_scores)
            flags = self._flag(dim_scores)

            self.results.append(
                EvaluationResult(
                    test_case_id=tc.id,
                    model_name=resp.model_name,
                    scores=dim_scores,
                    overall_score=overall,
                    flags=flags,
                )
            )
        return self.results

    @staticmethod
    def _flag(dim_scores) -> List[str]:
        flags = []
        if dim_scores["hallucination_risk"].score <= FLAG_THRESHOLDS["hallucination_risk"]:
            flags.append("possible_hallucination")
        if dim_scores["instruction_adherence"].score <= FLAG_THRESHOLDS["instruction_adherence"]:
            flags.append("instruction_miss")
        if dim_scores["reasoning_quality"].score <= FLAG_THRESHOLDS["reasoning_quality"]:
            flags.append("weak_reasoning")
        return flags

    # -- convenience aggregate views ------------------------------------------
    def leaderboard(self) -> List[Dict]:
        """Mean overall score per model, sorted best -> worst."""
        by_model: Dict[str, List[float]] = {}
        for r in self.results:
            by_model.setdefault(r.model_name, []).append(r.overall_score)
        rows = [
            {"model_name": m, "mean_overall_score": round(sum(v) / len(v), 3), "n": len(v)}
            for m, v in by_model.items()
        ]
        return sorted(rows, key=lambda r: r["mean_overall_score"], reverse=True)
