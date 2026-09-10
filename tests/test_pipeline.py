import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_eval import TestCase, ModelResponse, EvaluationPipeline, HeuristicScorer
from llm_eval.data_models import DIMENSIONS


def make_case():
    tc = TestCase(
        id="t1",
        prompt="Explain why the sky is blue.",
        reference_answer="The sky is blue due to Rayleigh scattering of sunlight by the atmosphere.",
        instructions=["Do not mention pollution", "Keep the answer under 3 sentences"],
    )
    good = ModelResponse("t1", "GoodModel", "The sky appears blue because of Rayleigh scattering: shorter blue wavelengths scatter more in the atmosphere than longer red wavelengths. This makes blue light reach our eyes from every direction.")
    bad = ModelResponse("t1", "BadModel", "According to a 2027 study, the sky is blue mostly because of pollution particles reflecting light, though I think it might also be about the ocean's reflection, not totally sure.")
    return tc, good, bad


def test_all_dimensions_scored():
    tc, good, _ = make_case()
    scorer = HeuristicScorer()
    scores = scorer.score(tc, good)
    assert set(scores.keys()) == set(DIMENSIONS)
    for ds in scores.values():
        assert 1 <= ds.score <= 5


def test_good_response_outscores_bad_response():
    tc, good, bad = make_case()
    pipeline = EvaluationPipeline(scorer=HeuristicScorer())
    results = pipeline.run([tc], [good, bad])
    scores = {r.model_name: r.overall_score for r in results}
    assert scores["GoodModel"] > scores["BadModel"]


def test_hallucination_flag_triggers_on_fabrication_markers():
    tc, _, bad = make_case()
    pipeline = EvaluationPipeline(scorer=HeuristicScorer())
    results = pipeline.run([tc], [bad])
    assert "possible_hallucination" in results[0].flags


def test_banned_phrase_instruction_is_detected_as_violated():
    tc, _, bad = make_case()
    pipeline = EvaluationPipeline(scorer=HeuristicScorer())
    results = pipeline.run([tc], [bad])
    assert results[0].scores["instruction_adherence"].score < 5


def test_leaderboard_is_sorted_descending():
    tc, good, bad = make_case()
    pipeline = EvaluationPipeline(scorer=HeuristicScorer())
    pipeline.run([tc], [good, bad])
    lb = pipeline.leaderboard()
    assert lb[0]["mean_overall_score"] >= lb[-1]["mean_overall_score"]


def test_missing_test_case_raises():
    pipeline = EvaluationPipeline(scorer=HeuristicScorer())
    orphan = ModelResponse("does_not_exist", "X", "irrelevant")
    try:
        pipeline.run([], [orphan])
        assert False, "expected ValueError"
    except ValueError:
        pass
