"""
Multi-dimensional scoring framework.

Two scorer implementations are provided behind a common interface
(BaseScorer):

1. LLMJudgeScorer   - uses an LLM (Anthropic Claude by default) as an
                       impartial judge, prompted with a standardized rubric
                       so every response is graded against the same
                       criteria. This is the recommended scorer for
                       production use.

2. HeuristicScorer  - a lightweight, API-free fallback used when no
                       judge model / API key is available (e.g. for demos,
                       CI, or offline testing). It approximates the same
                       five dimensions using text-similarity, length and
                       lexical heuristics so the rest of the pipeline can
                       be exercised end-to-end.

Both scorers return the same DimensionScore objects, so the rest of the
pipeline (evaluator.py, report.py) is agnostic to which one is used.
"""

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .data_models import DIMENSIONS, DIMENSION_WEIGHTS, DimensionScore, TestCase, ModelResponse


JUDGE_SYSTEM_PROMPT = """You are a strict, impartial evaluator of AI assistant responses.
Score the RESPONSE to the PROMPT on exactly these five dimensions, each on a 1-5 integer scale:

1. factual_accuracy: 5 = fully accurate, 1 = riddled with factual errors.
2. relevance: 5 = directly and completely addresses the prompt, 1 = off-topic.
3. reasoning_quality: 5 = logically sound, well-structured reasoning, 1 = incoherent or fallacious.
4. hallucination_risk: 5 = no fabricated facts/sources/entities, 1 = severe fabrication.
   (Note: 5 is GOOD here, i.e. this dimension is already inverted so higher is always better.)
5. instruction_adherence: 5 = follows every stated instruction/constraint, 1 = ignores them.

Use the REFERENCE ANSWER (if provided) only as a factual grounding aid, not as a style target.

Respond with ONLY a JSON object, no markdown fences, no preamble, in this exact shape:
{
  "factual_accuracy": {"score": <1-5>, "rationale": "<one short sentence>"},
  "relevance": {"score": <1-5>, "rationale": "<one short sentence>"},
  "reasoning_quality": {"score": <1-5>, "rationale": "<one short sentence>"},
  "hallucination_risk": {"score": <1-5>, "rationale": "<one short sentence>"},
  "instruction_adherence": {"score": <1-5>, "rationale": "<one short sentence>"}
}
"""


class BaseScorer(ABC):
    @abstractmethod
    def score(
        self, test_case: TestCase, response: ModelResponse
    ) -> Dict[str, DimensionScore]:
        ...

    @staticmethod
    def weighted_overall(scores: Dict[str, DimensionScore]) -> float:
        total = sum(scores[d].score * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
        return total  # already on a 1-5 scale since weights sum to 1


# ---------------------------------------------------------------------------
# LLM-as-judge scorer
# ---------------------------------------------------------------------------
class LLMJudgeScorer(BaseScorer):
    """Uses an LLM as an impartial judge against a standardized rubric.

    Works with the Anthropic Python SDK. Requires ANTHROPIC_API_KEY to be
    set in the environment. Falls back gracefully with a clear error if the
    SDK/key are missing -- callers should catch this and fall back to
    HeuristicScorer (see evaluator.py's `build_scorer` helper).
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. Run `pip install anthropic` "
                "or use HeuristicScorer instead."
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it, or use HeuristicScorer instead."
            )
        import anthropic
        self._client = anthropic.Anthropic()

    def score(self, test_case: TestCase, response: ModelResponse) -> Dict[str, DimensionScore]:
        user_content = (
            f"PROMPT:\n{test_case.prompt}\n\n"
            f"INSTRUCTIONS TO FOLLOW:\n{'; '.join(test_case.instructions) or 'none'}\n\n"
            f"REFERENCE ANSWER (optional grounding, may be blank):\n"
            f"{test_case.reference_answer or 'N/A'}\n\n"
            f"RESPONSE TO EVALUATE (from model '{response.model_name}'):\n"
            f"{response.response_text}"
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=500,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        raw_text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        raw_text = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(raw_text)

        scores = {}
        for dim in DIMENSIONS:
            d = parsed[dim]
            scores[dim] = DimensionScore(dimension=dim, score=float(d["score"]), rationale=d.get("rationale", ""))
        return scores


# ---------------------------------------------------------------------------
# Heuristic (offline / no-API-key) scorer
# ---------------------------------------------------------------------------
class HeuristicScorer(BaseScorer):
    """Approximate, API-free scorer used as a fallback / demo scorer.

    Not a substitute for LLM-as-judge grading, but lets the whole pipeline
    run deterministically without network access or API keys.
    """

    HEDGE_WORDS = {"maybe", "probably", "i think", "not sure", "possibly"}
    FABRICATION_MARKERS = {"according to a 2027 study", "as reported by", "cite:", "[source needed]"}

    def score(self, test_case: TestCase, response: ModelResponse) -> Dict[str, DimensionScore]:
        text = response.response_text
        text_lower = text.lower()

        scores = {
            "factual_accuracy": self._factual_accuracy(test_case, text_lower),
            "relevance": self._relevance(test_case, text_lower),
            "reasoning_quality": self._reasoning_quality(text),
            "hallucination_risk": self._hallucination_risk(test_case, text_lower),
            "instruction_adherence": self._instruction_adherence(test_case, text_lower),
        }
        return scores

    # -- individual heuristics -------------------------------------------------
    def _token_overlap(self, a: str, b: str) -> float:
        ta, tb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _factual_accuracy(self, tc: TestCase, text_lower: str) -> DimensionScore:
        if tc.reference_answer:
            overlap = self._token_overlap(tc.reference_answer, text_lower)
            score = 1 + round(overlap * 4)  # map [0,1] -> [1,5]
            rationale = f"Lexical overlap with reference answer: {overlap:.2f}"
        else:
            score = 3
            rationale = "No reference answer provided; neutral default score."
        return DimensionScore("factual_accuracy", float(score), rationale)

    def _relevance(self, tc: TestCase, text_lower: str) -> DimensionScore:
        overlap = self._token_overlap(tc.prompt, text_lower)
        score = 1 + round(min(overlap * 6, 4))
        return DimensionScore("relevance", float(score), f"Prompt/response keyword overlap: {overlap:.2f}")

    def _reasoning_quality(self, text: str) -> DimensionScore:
        sentences = [s for s in re.split(r"[.!?]", text) if s.strip()]
        connectors = len(re.findall(
            r"\b(because|therefore|thus|since|so that|as a result|so |=|:)\b", text.lower()
        ))
        length_score = min(len(sentences) / 2, 1.0)
        connector_score = min(connectors / 1, 1.0)
        score = 1 + round((0.5 * length_score + 0.5 * connector_score) * 4)
        return DimensionScore(
            "reasoning_quality", float(score),
            f"{len(sentences)} sentences, {connectors} logical connectors detected."
        )

    def _hallucination_risk(self, tc: TestCase, text_lower: str) -> DimensionScore:
        markers_found = [m for m in self.FABRICATION_MARKERS if m in text_lower]
        hedges = sum(1 for h in self.HEDGE_WORDS if h in text_lower)
        penalty = len(markers_found) * 2 + (1 if hedges >= 2 else 0)
        score = max(1, 5 - penalty)
        rationale = (
            f"Fabrication markers: {len(markers_found)}, hedge phrases: {hedges}."
        )
        return DimensionScore("hallucination_risk", float(score), rationale)

    def _instruction_adherence(self, tc: TestCase, text_lower: str) -> DimensionScore:
        """Checks each instruction using pattern-specific rules where possible
        (bullet counts, word limits, sentence limits, 'do not mention X'),
        falling back to keyword matching for open-ended instructions."""
        if not tc.instructions:
            return DimensionScore("instruction_adherence", 5.0, "No explicit instructions to check.")

        followed = 0
        details = []
        for instr in tc.instructions:
            ok, why = self._check_single_instruction(instr, text_lower)
            followed += int(ok)
            details.append(f"[{'ok' if ok else 'miss'}] {why}")

        ratio = followed / len(tc.instructions)
        score = 1 + round(ratio * 4)
        return DimensionScore(
            "instruction_adherence", float(score),
            f"{followed}/{len(tc.instructions)} instructions satisfied — " + "; ".join(details)
        )

    def _check_single_instruction(self, instr: str, text_lower: str):
        il = instr.lower()

        # "exactly N bullet points" / "N bullets"
        m = re.search(r"(exactly\s+)?(\d+)\s*bullet", il)
        if m:
            n = int(m.group(2))
            bullet_count = len(re.findall(r"^\s*[-*•]", text_lower, flags=re.MULTILINE))
            return bullet_count == n, f"expected {n} bullets, found {bullet_count}"

        # "no more than N words per bullet/line"
        m = re.search(r"no more than (\d+) words? per (bullet|line)", il)
        if m:
            limit = int(m.group(1))
            lines = [l for l in text_lower.splitlines() if l.strip()]
            over = [l for l in lines if len(l.split()) > limit]
            return len(over) == 0, f"word-per-line limit {limit}, {len(over)} line(s) over"

        # "under N sentences" / "no more than N sentences"
        m = re.search(r"(under|no more than|at most)\s+(\d+)\s+sentences?", il)
        if m:
            limit = int(m.group(2))
            n_sentences = len([s for s in re.split(r"[.!?]", text_lower) if s.strip()])
            return n_sentences <= limit, f"sentence limit {limit}, found {n_sentences}"

        # "do not mention X"
        m = re.search(r"do not mention (.+)", il)
        if m:
            banned = m.group(1).strip().rstrip(".")
            present = banned in text_lower
            return not present, f"banned phrase '{banned}' {'found' if present else 'absent'}"

        # "include both X and Y" / generic keyword fallback
        keywords = [k for k in re.findall(r"\w+", il) if len(k) > 3]
        hit = any(k in text_lower for k in keywords) if keywords else True
        return hit, f"keyword check on: {instr}"


def build_scorer(prefer_llm_judge: bool = True) -> BaseScorer:
    """Convenience factory: try LLM-as-judge, fall back to heuristic scorer."""
    if prefer_llm_judge:
        try:
            return LLMJudgeScorer()
        except RuntimeError:
            pass
    return HeuristicScorer()
