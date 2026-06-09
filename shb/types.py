"""Core data types — the contract between task families, agents, and the harness.

Three roles share these types:

* a **task family** generates :class:`TaskInstance` objects and grades
  :class:`Submission` objects into :class:`Grade` objects;
* an **agent** (the system under test) receives only an :class:`AgentView` and
  returns a :class:`Submission`;
* the **harness** generates instances, hands each agent a view, and grades.

Design points that close gaming channels (see docs/THREAT_MODEL.md):

* The view's ``task_id`` is an **opaque hash** — it carries no ``clean``/``trapped``
  label, seed, or twin information. The variant is never visible to the agent.
* Flaw reports are **structured and evidence-bearing** (:class:`Submission.issues`),
  drawn from a published controlled vocabulary (:func:`allowed_issue_kinds`). A
  bare keyword in free text does not count as detection.
* Agents report a **confidence**, so "confident wrong science" is measured, not
  inferred.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from .taxonomy import TrapType

VARIANTS = ("clean", "trapped")


def allowed_issue_kinds() -> list[str]:
    """The published controlled vocabulary an agent may use to flag a flaw."""
    return [t.value for t in TrapType]


def opaque_id(internal_id: str) -> str:
    """A stable, opaque public id that reveals neither variant nor seed.

    Mixing in ``SHB_SALT`` lets the private/official split use a salt unknown to
    submitters, so memorizing public ids buys nothing.
    """
    salt = os.environ.get("SHB_SALT", "")
    return "task_" + hashlib.sha256(f"{salt}|{internal_id}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class AgentView:
    """Exactly what the system under test is allowed to see.

    Excludes the variant, seed, ground-truth answer, and grading payload. The
    agent must discover any planted flaw from the data alone.
    """

    task_id: str                      # opaque hash; NOT the internal id
    domain: str
    family: str                       # the task *type* (does not reveal the trap)
    prompt: str
    assets: dict[str, str]
    answer_fields: dict[str, str]
    allowed_issue_kinds: list[str]    # controlled vocabulary for structured flaw reports


@dataclass(frozen=True)
class TaskInstance:
    """One concrete problem: a clean task or its trapped twin."""

    task_id: str                      # INTERNAL: "chem.theoretical_yield/seed=7/clean"
    family: str
    domain: str
    variant: str                      # "clean" | "trapped"
    seed: int
    prompt: str
    assets: dict[str, str]
    answer_fields: dict[str, str]
    trap_type: Optional[TrapType] = None
    trap_note: str = ""
    answer: dict[str, Any] = field(default_factory=dict)    # HIDDEN ground truth
    grading: dict[str, Any] = field(default_factory=dict)   # HIDDEN grader payload

    @property
    def is_trapped(self) -> bool:
        return self.variant == "trapped"

    @property
    def public_id(self) -> str:
        return opaque_id(self.task_id)

    def view(self, prompt_style: str = "cued") -> AgentView:
        prompt = self.prompt
        alt = self.grading.get("prompt_" + prompt_style)
        if alt:
            prompt = alt
        return AgentView(
            task_id=self.public_id, domain=self.domain, family=self.family,
            prompt=prompt, assets=dict(self.assets),
            answer_fields=dict(self.answer_fields),
            allowed_issue_kinds=allowed_issue_kinds(),
        )


@dataclass
class Submission:
    """What an agent returns for one task.

    ``issues`` is the structured flaw channel: a list of ``{"kind": <one of
    allowed_issue_kinds>, "evidence": <verifiable detail>}``. Detection requires
    naming the correct ``kind`` *and* supplying evidence the grader can check
    against ground truth (e.g. the offending ids, the detected unit, the
    non-converged residual). ``confidence`` in [0,1] is the agent's confidence in
    its ``answers``; it powers the confident-wrong-rate metric.
    """

    answers: dict[str, Any] = field(default_factory=dict)
    issues: list[dict] = field(default_factory=list)
    abstained: bool = False
    confidence: float = 1.0
    notes: str = ""

    def issues_of(self, kind: str) -> list[dict]:
        return [i for i in self.issues if isinstance(i, dict) and str(i.get("kind")) == kind]

    @property
    def has_any_issue(self) -> bool:
        return len(self.issues) > 0


@dataclass
class Grade:
    """The grader's verdict on one (instance, submission) pair."""

    task_id: str                      # internal id (for our reports; never shown to agents)
    family: str
    domain: str
    variant: str
    seed: int
    trap_type: Optional[str]
    paired: bool
    conclusion_correct: bool          # answer scientifically acceptable (corrected or correctly flagged)
    answer_correct: bool              # the *answer field* matches (ignoring detection)
    trap_detected: Optional[bool]     # trapped only: planted flaw flagged with valid evidence
    false_alarm: bool                 # claimed a flaw / abstained on a clean task
    confident_wrong: bool             # trapped: wrong answer, undetected, asserted with confidence
    confidence: float
    score: float
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    name: str

    def solve(self, view: AgentView, workdir: str) -> Submission: ...


@dataclass
class Family:
    """A generator of clean/trapped task pairs plus its grader and baselines.

    ``paired`` is True only when the clean and trapped instances for a seed are
    genuine counterfactual twins — identical in every respect except the injected
    flaw. Families whose variants differ in the underlying signal/sample (not a
    pure injection) set ``paired=False`` and are reported as separate robustness
    scenarios, never folded into the paired Fake-Science Gap.
    """

    family_id: str
    domain: str
    title: str
    description: str
    trap_types: list[TrapType]
    flaw_kind: str                    # the controlled-vocabulary kind a correct detection must name
    generate: Callable[[int, str], TaskInstance]
    grade: Callable[[TaskInstance, Submission], Grade]
    ref_naive: Callable[[AgentView], Submission]
    ref_careful: Callable[[AgentView], Submission]
    paired: bool = True
    variants: tuple[str, ...] = VARIANTS
