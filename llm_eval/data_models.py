"""
Data models for the LLM Response Evaluation & Benchmarking System.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Scoring dimensions
# ---------------------------------------------------------------------------
# Each dimension is scored on a 1-5 scale (5 = best) except hallucination_risk,
# which is scored 1-5 where 5 = LOW risk (i.e. already normalized so that
# "higher is always better" across every dimension for easy aggregation).
DIMENSIONS = [
    "factual_accuracy",
    "relevance",
    "reasoning_quality",
    "hallucination_risk",   # normalized: 5 = no hallucination, 1 = severe hallucination
    "instruction_adherence",
]

DIMENSION_WEIGHTS = {
    "factual_accuracy": 0.25,
    "relevance": 0.20,
    "reasoning_quality": 0.20,
    "hallucination_risk": 0.20,
    "instruction_adherence": 0.15,
}


@dataclass
class TestCase:
    """A single benchmark item: a prompt, optional reference answer, and
    optional instruction constraints the response must follow."""
    id: str
    prompt: str
    reference_answer: Optional[str] = None
    instructions: List[str] = field(default_factory=list)
    category: str = "general"


@dataclass
class ModelResponse:
    """One model's response to a given TestCase."""
    test_case_id: str
    model_name: str
    response_text: str


@dataclass
class DimensionScore:
    dimension: str
    score: float          # 1-5
    rationale: str = ""


@dataclass
class EvaluationResult:
    test_case_id: str
    model_name: str
    scores: Dict[str, DimensionScore]
    overall_score: float
    flags: List[str] = field(default_factory=list)   # e.g. "possible_hallucination"

    def to_row(self) -> dict:
        row = {
            "test_case_id": self.test_case_id,
            "model_name": self.model_name,
            "overall_score": round(self.overall_score, 3),
        }
        for dim, ds in self.scores.items():
            row[dim] = ds.score
        row["flags"] = "; ".join(self.flags)
        return row
