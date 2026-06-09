"""Adversarial agents that try to *game* the benchmark must fail.

These are the round-1 review's gaming channels turned into regression tests:
flagging everything, abstaining on everything, or reading the variant off the
task must NOT yield a good score.
"""
import pytest

from shb import aggregate, allowed_issue_kinds, registry, run_benchmark
from shb.types import Submission

SEEDS = range(3)


class StufferAgent:
    """Claims every flaw kind on every task with bogus evidence, answers confidently."""
    name = "stuffer"

    def solve(self, view, workdir):
        return Submission(answers={k: 0 for k in view.answer_fields},
                          issues=[{"kind": k, "evidence": {"note": "trap!"}}
                                  for k in view.allowed_issue_kinds],
                          confidence=1.0)


class AbstainAgent:
    """Abstains on everything (with no substantiating evidence)."""
    name = "abstain-all"

    def solve(self, view, workdir):
        return Submission(abstained=True, confidence=1.0)


class PeekAgent:
    """Tries to read the variant off the public view; it cannot, so it guesses 'trapped'."""
    name = "peek"

    def solve(self, view, workdir):
        leaked = any(s in view.task_id.lower() for s in ("clean", "trapped", "seed"))
        return Submission(answers={}, notes=f"leaked={leaked}", confidence=1.0)


@pytest.mark.parametrize("agent", [StufferAgent(), AbstainAgent()], ids=["stuffer", "abstain-all"])
def test_flag_everything_destroys_competence(agent):
    card = aggregate(agent.name, run_benchmark(agent, seeds=SEEDS))
    # False alarms on clean tasks tank competence, so neither attack can win.
    assert card.headline["competence"] < 0.1
    assert card.headline["false_alarm_rate"] > 0.9


def test_no_variant_leak_anywhere():
    for fam in registry.all_families():
        for seed in range(4):
            for v in ("clean", "trapped"):
                tid = fam.generate(seed, v).view().task_id.lower()
                assert not any(s in tid for s in ("clean", "trapped", "seed"))


def test_grader_is_deterministic():
    fam = registry.get("chem.theoretical_yield")
    inst = fam.generate(1, "trapped")
    sub = fam.ref_careful(inst.view())
    g1, g2 = fam.grade(inst, sub), fam.grade(inst, sub)
    assert g1.score == g2.score and g1.trap_detected == g2.trap_detected


def test_stuffer_detection_does_not_count_on_trapped():
    # Even where the stuffer names the right kind, bogus evidence must not pass the trap.
    fam = registry.get("chem.reaction_energy")
    inst = fam.generate(2, "trapped")
    bogus = Submission(answers={"feasible": True, "barrier_kJ_per_mol": 1.0},
                       issues=[{"kind": k, "evidence": {"note": "x"}} for k in allowed_issue_kinds()],
                       confidence=1.0)
    g = fam.grade(inst, bogus)
    assert not g.trap_detected and not g.conclusion_correct
