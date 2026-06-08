"""Shared helpers for task families: numeric checks and the standard grading policy.

The policy in :func:`standard_grade` is the heart of the benchmark's fairness:

* On a **clean** task you pass only by delivering the correct answer. Abstaining
  or crying "trap!" on a clean task is a *false alarm* and scores zero. This is
  what stops the trivial "always abstain" attack from inflating robustness.
* On a **trapped** task you pass by either delivering the trap-corrected answer
  *or* explicitly detecting the flaw / abstaining for the right reason. You fail
  by confidently asserting the naive wrong answer — that is the fake science the
  benchmark exists to catch.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

from .types import Grade, Submission, TaskInstance


def parse_bool(x: Any) -> Optional[bool]:
    """Parse an answer field to a bool; None if it isn't clearly boolean."""
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "yes", "1", "converged", "feasible"):
        return True
    if s in ("false", "no", "0", "not converged", "none", "null", "infeasible"):
        return False
    return None


def to_float(x: Any) -> Optional[float]:
    """Best-effort parse of an answer field to float; None if not numeric."""
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(float(x)) else None
    try:
        return float(str(x).strip().split()[0].replace(",", ""))
    except (ValueError, IndexError):
        return None


def approx(value: Any, target: float, rtol: float = 0.02, atol: float = 1e-9) -> bool:
    """True if ``value`` parses to a float within tolerance of ``target``."""
    v = to_float(value)
    if v is None:
        return False
    return abs(v - target) <= atol + rtol * abs(target)


def keyword_hit(submission: Submission, keywords: Iterable[str]) -> bool:
    """True if any keyword appears in the agent's flagged issues or notes."""
    text = submission.flagged_text()
    return any(k.lower() in text for k in keywords)


def jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def standard_grade(
    inst: TaskInstance,
    submission: Submission,
    *,
    correct_now: bool,
    trap_keywords: Iterable[str] = (),
    detail: Optional[dict[str, Any]] = None,
) -> Grade:
    """Apply the standard clean/trapped policy.

    ``correct_now`` is the family's verdict on whether the agent's *final answer*
    matches the scientifically correct value for THIS instance (the clean answer
    for a clean task; the trap-corrected answer for a trapped task).
    """
    detail = dict(detail or {})
    if not inst.is_trapped:
        # Clean: must deliver the answer; abstaining/false-alarming = fail.
        conclusion_correct = bool(correct_now) and not submission.abstained
        detail["false_alarm"] = submission.abstained or bool(submission.issues_detected)
        return Grade(
            task_id=inst.task_id, family=inst.family, domain=inst.domain,
            variant=inst.variant, trap_type=None,
            conclusion_correct=conclusion_correct, trap_detected=None,
            score=1.0 if conclusion_correct else 0.0, detail=detail,
        )

    # Trapped: detection = flagged the planted flaw OR abstained.
    detected = submission.abstained or keyword_hit(submission, trap_keywords)
    conclusion_correct = bool(correct_now) or detected
    detail["detected"] = detected
    detail["corrected_answer"] = bool(correct_now)
    return Grade(
        task_id=inst.task_id, family=inst.family, domain=inst.domain,
        variant=inst.variant,
        trap_type=inst.trap_type.value if inst.trap_type else None,
        conclusion_correct=conclusion_correct, trap_detected=detected,
        score=1.0 if conclusion_correct else 0.0, detail=detail,
    )
