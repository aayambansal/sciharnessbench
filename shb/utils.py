"""Shared helpers for task families: salted RNGs, numeric checks, grading policy.

The v2 grading policy (see :func:`standard_grade`) closes the gaming channels the
round-1 review found:

* On a **clean** task you pass only by delivering the correct answer with **no**
  flaw claim and **no** abstention. Any issue or abstention is a *false alarm* and
  scores zero — so "always flag" / "always abstain" destroys competence.
* On a **trapped** task you pass by delivering the trap-corrected answer, or by
  raising a **structured issue of the correct kind with verifiable evidence**
  (equivalently, abstaining with that evidence). A bare keyword, a wrong-kind
  flag, or evidence that does not check out does **not** count — so keyword
  stuffing fails.
* A **confident wrong** answer on a trapped task (wrong, undetected, asserted with
  confidence >= 0.5) is recorded explicitly: that is the fake-science event.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import re
from typing import Any, Callable, Iterable, Optional

from .types import Grade, Submission, TaskInstance


def _salt() -> str:
    return os.environ.get("SHB_SALT", "")


def family_rng(family_id: str, seed: int, trap: bool = False) -> random.Random:
    """Deterministic, salted RNG. Base and trap streams are independent.

    Mixing in SHB_SALT yields the private/official instance set; the public
    (salt-free) instances are a transparent dev split.
    """
    key = f"{_salt()}|{family_id}|{seed}|{'trap' if trap else 'base'}"
    return random.Random(key)


def np_seed(family_id: str, seed: int, trap: bool = False) -> int:
    key = f"{_salt()}|{family_id}|{seed}|{'trap' if trap else 'base'}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def base_signature(*parts: Any) -> str:
    """Hash of the non-trap base of an instance; paired twins must share it."""
    return hashlib.sha256("|".join(map(repr, parts)).encode()).hexdigest()[:16]


def parse_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "yes", "1", "converged", "feasible", "stable"):
        return True
    if s in ("false", "no", "0", "not converged", "none", "null", "infeasible", "unstable"):
        return False
    return None


def to_float(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(float(x)) else None
    try:
        return float(str(x).strip().split()[0].replace(",", ""))
    except (ValueError, IndexError):
        return None


def approx(value: Any, target: float, rtol: float = 0.02, atol: float = 1e-9) -> bool:
    v = to_float(value)
    return v is not None and abs(v - target) <= atol + rtol * abs(target)


def _safe(fn: Callable, x: Any) -> bool:
    try:
        return bool(fn(x))
    except Exception:
        return False


def _paired(family_id: str) -> bool:
    from . import registry
    fam = registry._REGISTRY.get(family_id)
    return fam.paired if fam else True


def standard_grade(
    inst: TaskInstance,
    sub: Submission,
    *,
    answer_correct: bool,
    flaw_kind: str,
    evidence_ok: Callable[[Any], bool],
    detail: Optional[dict[str, Any]] = None,
) -> Grade:
    """Apply the v2 clean/trapped policy.

    ``answer_correct`` — the family's verdict on whether the *answer fields* match
    the scientifically correct value for THIS instance (clean answer for a clean
    task; trap-corrected answer for a trapped task).
    ``flaw_kind`` — the controlled-vocabulary kind a correct detection must name.
    ``evidence_ok`` — verifies an issue's evidence against ground truth.
    """
    detail = dict(detail or {})
    conf = max(0.0, min(1.0, float(sub.confidence)))
    detected = any(_safe(evidence_ok, i.get("evidence")) for i in sub.issues_of(flaw_kind))
    paired = _paired(inst.family)
    detail.update(detected=detected, answer_correct=bool(answer_correct))

    if not inst.is_trapped:
        false_alarm = sub.has_any_issue or sub.abstained
        ok = bool(answer_correct) and not false_alarm
        return Grade(
            task_id=inst.task_id, family=inst.family, domain=inst.domain,
            variant=inst.variant, seed=inst.seed, trap_type=None, paired=paired,
            conclusion_correct=ok, answer_correct=bool(answer_correct),
            trap_detected=None, false_alarm=false_alarm, confident_wrong=False,
            confidence=conf, score=1.0 if ok else 0.0, detail=detail)

    ok = bool(answer_correct) or detected
    confident_wrong = (not answer_correct) and (not detected) and conf >= 0.5
    return Grade(
        task_id=inst.task_id, family=inst.family, domain=inst.domain,
        variant=inst.variant, seed=inst.seed,
        trap_type=inst.trap_type.value if inst.trap_type else None,
        paired=paired, conclusion_correct=ok, answer_correct=bool(answer_correct),
        trap_detected=detected, false_alarm=False, confident_wrong=confident_wrong,
        confidence=conf, score=1.0 if ok else 0.0, detail=detail)


def issue(kind: str, **evidence: Any) -> dict:
    """Build a structured flaw report for a Submission."""
    return {"kind": kind, "evidence": dict(evidence)}


# --- evidence parsing, shared by every family's evidence_ok check -------------
def ev_text(ev: Any) -> str:
    if isinstance(ev, dict):
        return " ".join(f"{k} {v}" for k, v in ev.items()).lower()
    if isinstance(ev, (list, tuple, set)):
        return " ".join(map(str, ev)).lower()
    return str(ev).lower()


def ev_numbers(ev: Any) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d*\.\d+|-?\d+", ev_text(ev))]


def ev_contains(ev: Any, *subs: str) -> bool:
    t = ev_text(ev)
    return all(s.lower() in t for s in subs)


def ev_near(ev: Any, target: float, rtol: float = 0.05, atol: float = 1e-9) -> bool:
    return any(abs(n - target) <= atol + rtol * abs(target) for n in ev_numbers(ev))
