"""Reference baseline agents.

These are not the systems under test — they are oracles that prove the benchmark
works and document, per family, exactly what "fake science" vs "real science"
looks like:

* ``ReferenceAgent("naive")`` does the obvious computation, trusts every input
  at face value, and reports the last number it produced. It should pass clean
  tasks and *fail* trapped ones — the fake-science signature.
* ``ReferenceAgent("careful")`` runs the validation a competent scientist would
  (parse and sanity-check inputs, check units, check convergence, recompute from
  primary data, look for confounds) and either corrects or flags the flaw. It
  should pass *both* clean and trapped tasks.

The per-family logic lives in each family's ``ref_naive`` / ``ref_careful``,
which operate only on the public :class:`AgentView` — they get no hidden ground
truth, so they are honest baselines, not cheaters.
"""
from __future__ import annotations

from .. import registry
from ..types import AgentView, Submission


class ReferenceAgent:
    def __init__(self, mode: str):
        if mode not in ("naive", "careful"):
            raise ValueError("mode must be 'naive' or 'careful'")
        self.mode = mode
        self.name = f"reference-{mode}"

    def solve(self, view: AgentView, workdir: str) -> Submission:
        fam = registry.get(view.family)
        fn = fam.ref_careful if self.mode == "careful" else fam.ref_naive
        return fn(view)
