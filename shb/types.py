"""Core data types — the contract between task families, agents, and the harness.

Three roles share these types:

* a **task family** generates :class:`TaskInstance` objects and grades
  :class:`Submission` objects into :class:`Grade` objects;
* an **agent** (the system under test) receives only an :class:`AgentView` and
  returns a :class:`Submission`;
* the **harness** generates instances, hands each agent a view, and grades.

Nothing here imports a scientific library, so the contract stays cheap to
import and easy to reason about. Domain code lives under :mod:`shb.domains`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from .taxonomy import TrapType

VARIANTS = ("clean", "trapped")


@dataclass(frozen=True)
class AgentView:
    """Exactly what the system under test is allowed to see.

    Crucially this excludes ``answer`` and ``grading`` — the hidden ground
    truth — and the ``trap_type``. The agent must discover any planted flaw on
    its own, the way a scientist works from the data in front of them.
    """

    task_id: str
    family: str
    domain: str
    prompt: str
    assets: dict[str, str]            # filename -> file contents
    answer_fields: dict[str, str]     # field name -> description of expected value


@dataclass(frozen=True)
class TaskInstance:
    """One concrete problem: a clean task or its trapped twin."""

    task_id: str                      # "chem.theoretical_yield/seed=7/clean"
    family: str                       # "chem.theoretical_yield"
    domain: str                       # "chemistry"
    variant: str                      # "clean" | "trapped"
    seed: int
    prompt: str
    assets: dict[str, str]
    answer_fields: dict[str, str]
    trap_type: Optional[TrapType] = None
    trap_note: str = ""               # what the planted flaw is (for reports, not the agent)
    answer: dict[str, Any] = field(default_factory=dict)    # HIDDEN ground truth
    grading: dict[str, Any] = field(default_factory=dict)   # HIDDEN grader payload

    @property
    def is_trapped(self) -> bool:
        return self.variant == "trapped"

    def view(self) -> AgentView:
        return AgentView(
            task_id=self.task_id,
            family=self.family,
            domain=self.domain,
            prompt=self.prompt,
            assets=dict(self.assets),
            answer_fields=dict(self.answer_fields),
        )


@dataclass
class Submission:
    """What an agent returns for one task.

    ``answers`` holds the requested fields. ``issues_detected`` is the standard
    channel a careful agent uses to flag planted flaws ("molecular weight in
    table disagrees with structure", "SCF did not converge"); graders match
    these against family keywords. ``abstained`` says the agent judged the task
    unanswerable as posed — a legitimate, correct response to many traps.
    """

    answers: dict[str, Any] = field(default_factory=dict)
    issues_detected: list[str] = field(default_factory=list)
    abstained: bool = False
    notes: str = ""

    def flagged_text(self) -> str:
        return (" ".join(self.issues_detected) + " " + self.notes).lower()


@dataclass
class Grade:
    """The grader's verdict on one (instance, submission) pair."""

    task_id: str
    family: str
    domain: str
    variant: str
    trap_type: Optional[str]
    conclusion_correct: bool          # final answer scientifically acceptable
    trap_detected: Optional[bool]     # trapped only: planted flaw flagged? None for clean
    score: float                      # 0..1 instance score
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    """The system under test. Implement :meth:`solve`.

    ``solve`` receives the public view and a scratch ``workdir`` (the assets are
    also written there as files) and returns a :class:`Submission`.
    """

    name: str

    def solve(self, view: AgentView, workdir: str) -> Submission: ...


@dataclass
class Family:
    """A generator of clean/trapped task pairs plus its grader and baselines.

    ``generate(seed, variant)`` must be deterministic in ``seed``: same seed and
    variant => identical instance. The clean and trapped twins for a given seed
    should differ *only* by the planted trap, so the gap isolates the trap.
    """

    family_id: str
    domain: str
    title: str
    description: str
    trap_types: list[TrapType]
    generate: Callable[[int, str], TaskInstance]          # (seed, variant) -> instance
    grade: Callable[[TaskInstance, Submission], Grade]
    ref_naive: Callable[[AgentView], Submission]          # trusts inputs -> fake science
    ref_careful: Callable[[AgentView], Submission]        # validates -> real science
    variants: tuple[str, ...] = VARIANTS
