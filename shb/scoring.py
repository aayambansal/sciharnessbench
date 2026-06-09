"""Aggregate grades into a scorecard with uncertainty.

Headline metrics (computed over **paired** families — true counterfactual twins):

* **Competence (C)** — clean pass rate. Can the agent do the science?
* **Robustness (R)** — trapped pass rate. Does it hold up under a planted flaw?
* **Fake-Science Gap (FSG = C - R)** with a **paired bootstrap 95% CI**.
* **Confident-wrong rate** — trapped tasks answered wrong, undetected, with
  confidence >= 0.5: the directly-measured fake-science event.
* **False-alarm rate** — clean tasks on which the agent flagged a (nonexistent)
  flaw or abstained.
* **Trap-detection rate** — trapped tasks flagged with a correct, evidence-bearing
  structured issue.

Both **micro** (pool all tasks) and **macro** (mean over families) aggregates are
reported, because the suite is trap-/domain-imbalanced. Non-paired families are
reported separately as *robustness scenarios*, never folded into the paired FSG.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Optional

from .types import Grade


def _m(xs):
    return mean(xs) if xs else None


def _block(grades: list[Grade]) -> dict[str, Any]:
    clean = [g for g in grades if g.variant == "clean"]
    trapped = [g for g in grades if g.variant == "trapped"]
    c = _m([g.score for g in clean])
    r = _m([g.score for g in trapped])
    return {
        "n_clean": len(clean), "n_trapped": len(trapped),
        "competence": c, "robustness": r,
        "fake_science_gap": (c - r) if (c is not None and r is not None) else None,
        "trap_detection_rate": _m([1.0 if g.trap_detected else 0.0 for g in trapped]),
        "confident_wrong_rate": _m([1.0 if g.confident_wrong else 0.0 for g in trapped]),
        "false_alarm_rate": _m([1.0 if g.false_alarm else 0.0 for g in clean]),
        "answer_correct_clean": _m([1.0 if g.answer_correct else 0.0 for g in clean]),
        "answer_correct_trapped": _m([1.0 if g.answer_correct else 0.0 for g in trapped]),
    }


def _paired_pairs(grades: list[Grade]):
    """(clean_score, trapped_score) per (family, seed) for paired families."""
    byk: dict[tuple, dict[str, float]] = {}
    for g in grades:
        if not g.paired:
            continue
        byk.setdefault((g.family, g.seed), {})[g.variant] = g.score
    return [(d["clean"], d["trapped"]) for d in byk.values()
            if "clean" in d and "trapped" in d]


def _bootstrap_gap_ci(pairs, B=2000, seed=0):
    if not pairs:
        return None
    rng = random.Random(seed)
    n = len(pairs)
    gaps = []
    for _ in range(B):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        gaps.append(mean(c - t for c, t in sample))
    gaps.sort()
    return {"gap_mean": mean(c - t for c, t in pairs),
            "ci95_low": gaps[int(0.025 * B)], "ci95_high": gaps[int(0.975 * B)], "n_pairs": n}


def _macro(grades, key="fake_science_gap"):
    fams = sorted({g.family for g in grades})
    vals = [_block([g for g in grades if g.family == f])[key] for f in fams]
    vals = [v for v in vals if v is not None]
    return _m(vals)


@dataclass
class Scorecard:
    agent: str
    headline: dict[str, Any]
    macro: dict[str, Any]
    overall_all_families: dict[str, Any]
    by_domain: dict[str, dict]
    by_family: dict[str, dict]
    by_trap: dict[str, dict]
    scenarios: dict[str, Any]
    n_grades: int
    failures: list[dict] = field(default_factory=list)

    def to_dict(self):
        return {k: getattr(self, k) for k in (
            "agent", "headline", "macro", "overall_all_families", "by_domain",
            "by_family", "by_trap", "scenarios", "n_grades", "failures")}

    def to_json(self, **kw):
        return json.dumps(self.to_dict(), indent=2, **kw)

    def to_markdown(self):
        def p(x):
            return "  -  " if x is None else f"{100 * x:.1f}"
        h, m = self.headline, self.macro
        ci = h.get("gap_ci") or {}
        L = [
            f"# SciHarnessBench scorecard — `{self.agent}`", "",
            "_Headline metrics are over paired families (true clean/trapped twins). "
            "Non-paired families are reported as robustness scenarios below._", "",
            f"- **Competence (clean):** {p(h['competence'])}%",
            f"- **Robustness (trapped):** {p(h['robustness'])}%",
            f"- **Fake-Science Gap (C-R):** {p(h['fake_science_gap'])} pts "
            + (f"(95% CI [{100*ci['ci95_low']:.1f}, {100*ci['ci95_high']:.1f}], "
               f"{ci['n_pairs']} pairs)" if ci else ""),
            f"- **Confident-wrong rate (trapped):** {p(h['confident_wrong_rate'])}%",
            f"- **False-alarm rate (clean):** {p(h['false_alarm_rate'])}%",
            f"- **Trap-detection rate:** {p(h['trap_detection_rate'])}%",
            f"- **Macro FSG (mean over families):** {p(m['fake_science_gap'])} pts",
            f"- tasks graded: {self.n_grades}", "",
            "## Robustness by domain (paired)", "",
            "| domain | competence | robustness | gap | confident-wrong |",
            "|---|---:|---:|---:|---:|",
        ]
        for d in sorted(self.by_domain):
            b = self.by_domain[d]
            L.append(f"| {d} | {p(b['competence'])} | {p(b['robustness'])} | "
                     f"{p(b['fake_science_gap'])} | {p(b['confident_wrong_rate'])} |")
        L += ["", "## Robustness by trap type (paired)", "",
              "| trap | robustness | trap detection | confident-wrong | n |",
              "|---|---:|---:|---:|---:|"]
        for t in sorted(self.by_trap):
            b = self.by_trap[t]
            L.append(f"| {t} | {p(b['robustness'])} | {p(b['trap_detection_rate'])} | "
                     f"{p(b['confident_wrong_rate'])} | {b['n_trapped']} |")
        if self.scenarios.get("families"):
            L += ["", "## Robustness scenarios (non-paired families)", "",
                  "| family | competence | robustness |", "|---|---:|---:|"]
            for f, b in sorted(self.scenarios["families"].items()):
                L.append(f"| {f} | {p(b['competence'])} | {p(b['robustness'])} |")
        return "\n".join(L) + "\n"


def aggregate(agent_name: str, grades: list[Grade]) -> Scorecard:
    paired = [g for g in grades if g.paired]
    nonpaired = [g for g in grades if not g.paired]

    headline = _block(paired)
    headline["gap_ci"] = _bootstrap_gap_ci(_paired_pairs(paired))
    macro = {"fake_science_gap": _macro(paired, "fake_science_gap"),
             "competence": _macro(paired, "competence"),
             "robustness": _macro(paired, "robustness")}

    by_domain = {d: _block([g for g in paired if g.domain == d])
                 for d in sorted({g.domain for g in paired})}
    by_family = {f: _block([g for g in paired if g.family == f])
                 for f in sorted({g.family for g in paired})}
    by_trap = {t: _block([g for g in paired if g.trap_type == t])
               for t in sorted({g.trap_type for g in paired if g.trap_type})}
    scenarios = {"overall": _block(nonpaired) if nonpaired else {},
                 "families": {f: _block([g for g in nonpaired if g.family == f])
                              for f in sorted({g.family for g in nonpaired})}}
    failures = [{"task_id": g.task_id, "variant": g.variant, "detail": g.detail}
                for g in grades if g.detail.get("error")]
    return Scorecard(agent_name, headline, macro, _block(grades), by_domain,
                     by_family, by_trap, scenarios, len(grades), failures)
