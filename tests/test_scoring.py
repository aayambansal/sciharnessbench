"""The v2 grading policy and scorecard aggregation."""
from shb.scoring import aggregate
from shb.taxonomy import TrapType
from shb.types import Grade, Submission, TaskInstance
from shb.utils import issue, standard_grade

OK = lambda ev: True   # noqa: E731 — permissive evidence check for policy tests


def _inst(variant, trap=None):
    return TaskInstance(task_id=f"f/seed=0/{variant}", family="f", domain="d", variant=variant,
                        seed=0, prompt="", assets={}, answer_fields={}, trap_type=trap)


def test_clean_correct_passes():
    g = standard_grade(_inst("clean"), Submission(answers={"x": 1}),
                       answer_correct=True, flaw_kind="unit_mismatch", evidence_ok=OK)
    assert g.conclusion_correct and g.score == 1.0 and not g.false_alarm


def test_clean_with_issue_is_false_alarm():
    g = standard_grade(_inst("clean"), Submission(issues=[issue("unit_mismatch", x=1)]),
                       answer_correct=True, flaw_kind="unit_mismatch", evidence_ok=OK)
    assert g.false_alarm and not g.conclusion_correct and g.score == 0.0


def test_clean_abstain_is_false_alarm():
    g = standard_grade(_inst("clean"), Submission(abstained=True),
                       answer_correct=True, flaw_kind="unit_mismatch", evidence_ok=OK)
    assert g.false_alarm and not g.conclusion_correct


def test_trapped_corrected_passes():
    g = standard_grade(_inst("trapped", TrapType.UNIT_MISMATCH), Submission(answers={"x": 1}),
                       answer_correct=True, flaw_kind="unit_mismatch", evidence_ok=OK)
    assert g.conclusion_correct and not g.confident_wrong


def test_trapped_detected_with_valid_evidence_passes():
    g = standard_grade(_inst("trapped", TrapType.UNIT_MISMATCH),
                       Submission(issues=[issue("unit_mismatch", unit="kcal")]),
                       answer_correct=False, flaw_kind="unit_mismatch",
                       evidence_ok=lambda ev: "kcal" in str(ev).lower())
    assert g.conclusion_correct and g.trap_detected and not g.confident_wrong


def test_trapped_wrong_kind_does_not_count():
    g = standard_grade(_inst("trapped", TrapType.UNIT_MISMATCH),
                       Submission(issues=[issue("decoy_data", x=1)], confidence=1.0),
                       answer_correct=False, flaw_kind="unit_mismatch", evidence_ok=OK)
    assert not g.conclusion_correct and not g.trap_detected and g.confident_wrong


def test_trapped_bad_evidence_does_not_count():
    g = standard_grade(_inst("trapped", TrapType.UNIT_MISMATCH),
                       Submission(issues=[issue("unit_mismatch", unit="parsecs")], confidence=1.0),
                       answer_correct=False, flaw_kind="unit_mismatch",
                       evidence_ok=lambda ev: "kcal" in str(ev).lower())
    assert not g.conclusion_correct and not g.trap_detected and g.confident_wrong


def test_trapped_confident_wrong_recorded():
    g = standard_grade(_inst("trapped", TrapType.UNIT_MISMATCH),
                       Submission(answers={"x": 2}, confidence=0.9),
                       answer_correct=False, flaw_kind="unit_mismatch", evidence_ok=OK)
    assert g.confident_wrong and g.score == 0.0


def test_aggregate_headline_paired_only():
    grades = [
        Grade("f/seed=0/clean", "f", "d", "clean", 0, None, True, True, True, None, False, False, 1.0, 1.0),
        Grade("f/seed=0/trapped", "f", "d", "trapped", 0, "unit_mismatch", True, False, False, False, False, True, 1.0, 0.0),
        # a non-paired scenario grade must be excluded from the headline
        Grade("g/seed=0/trapped", "g", "d", "trapped", 0, "confounding", False, True, True, True, False, False, 1.0, 1.0),
    ]
    card = aggregate("x", grades)
    assert card.headline["competence"] == 1.0
    assert card.headline["robustness"] == 0.0      # only the paired trapped grade counts
    assert card.headline["fake_science_gap"] == 1.0
    assert card.headline["confident_wrong_rate"] == 1.0
    assert "g" in card.scenarios["families"]        # non-paired reported separately
