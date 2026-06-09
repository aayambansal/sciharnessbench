"""SciHarnessBench — the fake-science benchmark for AI science agents.

Measures whether a scientific agent produces confident-but-wrong results when
data contains a flaw a competent scientist would catch. Every task ships as a
clean/trapped pair; the headline metric is the gap between them. Fully
self-generating and self-grading: no human in the loop, at authoring or eval.
"""
from __future__ import annotations

from . import registry
from .runner import run_benchmark, run_family, run_instance
from .scoring import Scorecard, aggregate
from .taxonomy import TRAP_META, TrapType
from .types import (Agent, AgentView, Family, Grade, Submission, TaskInstance,
                    allowed_issue_kinds, opaque_id)
from .utils import (approx, base_signature, family_rng, issue, np_seed, parse_bool,
                    standard_grade, to_float)

__version__ = "0.2.0"

__all__ = [
    "registry", "run_benchmark", "run_family", "run_instance",
    "Scorecard", "aggregate", "TRAP_META", "TrapType",
    "Agent", "AgentView", "Family", "Grade", "Submission", "TaskInstance",
    "allowed_issue_kinds", "opaque_id", "approx", "base_signature", "family_rng",
    "issue", "np_seed", "parse_bool", "standard_grade", "to_float",
]
