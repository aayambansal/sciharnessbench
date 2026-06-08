"""The grading policy and scorecard aggregation."""
from shb.scoring import aggregate
from shb.taxonomy import TrapType
from shb.types import Grade, Submission, TaskInstance
from shb.utils import standard_grade


def _inst(variant, trap=None):
    return TaskInstance(task_id=f"f/{variant}", family="f", domain="d", variant=variant,
                        seed=0, prompt="", assets={}, answer_fields={}, trap_type=trap)


def test_clean_correct_passes():
    g = standard_grade(_inst("clean"), Submission(answers={"x": 1}), correct_now=True)
    assert g.conclusion_correct and g.score == 1.0 and g.trap_detected is None


def test_clean_abstention_is_a_false_alarm():
    # Crying "trap!" on a clean task must NOT pass — this defeats the always-abstain attack.
    g = standard_grade(_inst("clean"), Submission(abstained=True), correct_now=True)
    assert not g.conclusion_correct and g.score == 0.0


def test_trapped_corrected_answer_passes():
    g = standard_grade(_inst("trapped", TrapType.UNIT_MISMATCH),
                       Submission(answers={"x": 1}), correct_now=True, trap_keywords=["unit"])
    assert g.conclusion_correct


def test_trapped_flagged_passes():
    g = standard_grade(_inst("trapped", TrapType.UNIT_MISMATCH),
                       Submission(issues_detected=["unit mismatch: kcal vs kJ"]),
                       correct_now=False, trap_keywords=["unit"])
    assert g.conclusion_correct and g.trap_detected


def test_trapped_abstain_passes():
    g = standard_grade(_inst("trapped", TrapType.UNIT_MISMATCH),
                       Submission(abstained=True), correct_now=False, trap_keywords=["unit"])
    assert g.conclusion_correct and g.trap_detected


def test_trapped_confident_wrong_fails():
    # The fake-science signature: confident wrong answer, no flag.
    g = standard_grade(_inst("trapped", TrapType.UNIT_MISMATCH),
                       Submission(answers={"x": 2}), correct_now=False, trap_keywords=["unit"])
    assert not g.conclusion_correct and not g.trap_detected and g.score == 0.0


def test_aggregate_metrics():
    grades = [
        Grade("a", "f", "d", "clean", None, True, None, 1.0),
        Grade("b", "f", "d", "clean", None, False, None, 0.0),
        Grade("c", "f", "d", "trapped", "unit_mismatch", False, False, 0.0),
        Grade("e", "f", "d", "trapped", "unit_mismatch", True, True, 1.0),
    ]
    card = aggregate("x", grades)
    o = card.overall
    assert o["competence"] == 0.5
    assert o["robustness"] == 0.5
    assert o["fake_science_gap"] == 0.0
    assert o["trap_detection_rate"] == 0.5
