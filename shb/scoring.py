"""Aggregate grades into a scorecard.

Headline metrics:

* **Competence (C)** — pass rate on clean tasks. Can the agent do the science?
* **Robustness (R)** — pass rate on trapped tasks. Does it hold up when the data
  is flawed?
* **Fake-Science Gap (FSG = C - R)** — the headline. How much does a planted
  flaw degrade the agent? An agent doing real science has FSG ~ 0; an agent
  doing fake science has high C, low R, large FSG.
* **Trap Detection Rate (TDR)** — fraction of trapped tasks where the agent
  explicitly flagged the planted flaw.

C and R are reported together on purpose: a high R alone is meaningless (you can
get it by abstaining on everything), but abstaining tanks C. The target agent is
high-C *and* high-R.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Optional

from .types import Grade


def _mean(xs: list[float]) -> Optional[float]:
    return mean(xs) if xs else None


def _block(grades: list[Grade]) -> dict[str, Any]:
    clean = [g for g in grades if g.variant == "clean"]
    trapped = [g for g in grades if g.variant == "trapped"]
    c = _mean([g.score for g in clean])
    r = _mean([g.score for g in trapped])
    tdr = _mean([1.0 if g.trap_detected else 0.0 for g in trapped if g.trap_detected is not None])
    return {
        "n_clean": len(clean),
        "n_trapped": len(trapped),
        "competence": c,
        "robustness": r,
        "fake_science_gap": (c - r) if (c is not None and r is not None) else None,
        "trap_detection_rate": tdr,
    }


@dataclass
class Scorecard:
    agent: str
    overall: dict[str, Any]
    by_domain: dict[str, dict[str, Any]]
    by_family: dict[str, dict[str, Any]]
    by_trap: dict[str, dict[str, Any]]
    n_grades: int
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent, "n_grades": self.n_grades,
            "overall": self.overall, "by_domain": self.by_domain,
            "by_family": self.by_family, "by_trap": self.by_trap,
            "failures": self.failures,
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), indent=2, **kw)

    def to_markdown(self) -> str:
        def pct(x: Optional[float]) -> str:
            return "  -  " if x is None else f"{100 * x:5.1f}"
        o = self.overall
        L = [
            f"# SciHarnessBench scorecard — `{self.agent}`",
            "",
            f"- **Competence (clean pass rate):** {pct(o['competence'])}%",
            f"- **Robustness (trapped pass rate):** {pct(o['robustness'])}%",
            f"- **Fake-Science Gap (C - R):** {pct(o['fake_science_gap'])} pts  "
            f"_(lower = less fake science; ~0 is ideal)_",
            f"- **Trap Detection Rate:** {pct(o['trap_detection_rate'])}%",
            f"- tasks graded: {self.n_grades}",
            "",
            "## By domain",
            "",
            "| domain | competence | robustness | fake-science gap | trap detection |",
            "|---|---:|---:|---:|---:|",
        ]
        for dom in sorted(self.by_domain):
            b = self.by_domain[dom]
            L.append(f"| {dom} | {pct(b['competence'])} | {pct(b['robustness'])} "
                     f"| {pct(b['fake_science_gap'])} | {pct(b['trap_detection_rate'])} |")
        L += ["", "## By trap type", "",
              "| trap | robustness | trap detection | n |", "|---|---:|---:|---:|"]
        for trap in sorted(self.by_trap):
            b = self.by_trap[trap]
            L.append(f"| {trap} | {pct(b['robustness'])} | {pct(b['trap_detection_rate'])} "
                     f"| {b['n_trapped']} |")
        L += ["", "## By family", "",
              "| family | competence | robustness | gap | trap detection |",
              "|---|---:|---:|---:|---:|"]
        for fam in sorted(self.by_family):
            b = self.by_family[fam]
            L.append(f"| {fam} | {pct(b['competence'])} | {pct(b['robustness'])} "
                     f"| {pct(b['fake_science_gap'])} | {pct(b['trap_detection_rate'])} |")
        return "\n".join(L) + "\n"


def aggregate(agent_name: str, grades: list[Grade]) -> Scorecard:
    by_domain: dict[str, dict[str, Any]] = {}
    by_family: dict[str, dict[str, Any]] = {}
    by_trap: dict[str, dict[str, Any]] = {}

    for dom in sorted({g.domain for g in grades}):
        by_domain[dom] = _block([g for g in grades if g.domain == dom])
    for fam in sorted({g.family for g in grades}):
        by_family[fam] = _block([g for g in grades if g.family == fam])
    for trap in sorted({g.trap_type for g in grades if g.trap_type}):
        by_trap[trap] = _block([g for g in grades if g.trap_type == trap])

    failures = [
        {"task_id": g.task_id, "variant": g.variant, "trap_type": g.trap_type,
         "detail": g.detail}
        for g in grades if g.detail.get("error")
    ]
    return Scorecard(
        agent=agent_name, overall=_block(grades),
        by_domain=by_domain, by_family=by_family, by_trap=by_trap,
        n_grades=len(grades), failures=failures,
    )
