"""Aggregate grades into a scorecard with behavior-separated metrics and uncertainty.

Headline metrics (over **paired** families — true clean/trapped twins):

* **Competence (C)** — clean pass rate.
* **Robustness (R)** — trapped pass rate (corrected answer *or* verified detection).
* **Fake-Science Gap (FSG = C - R)** with a **cluster bootstrap 95% CI** (resampling
  whole families, not iid twins, to respect within-family correlation).

Because a single robustness number mixes distinct behaviors, we also report, on
trapped tasks, the **corrected-answer rate** (fixed the answer), the
**detection rate** (flagged the flaw with verified evidence), the
**wrong-undetected rate** (fake science, regardless of confidence), the
**confident-wrong rate** (wrong-undetected at confidence >= 0.5), and the
**crash rate** (unsafe failures). The **false-alarm rate** is reported on clean
tasks. Everything is given micro (pooled) and macro (mean over families); the
non-paired families get their own **scenario headline**, never folded into the FSG.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Optional


def _m(xs):
    return mean(xs) if xs else None


def _block(grades) -> dict[str, Any]:
    clean = [g for g in grades if g.variant == "clean"]
    trapped = [g for g in grades if g.variant == "trapped"]
    c, r = _m([g.score for g in clean]), _m([g.score for g in trapped])
    return {
        "n_clean": len(clean), "n_trapped": len(trapped),
        "competence": c, "robustness": r,
        "fake_science_gap": (c - r) if (c is not None and r is not None) else None,
        "trap_detection_rate": _m([1.0 if g.trap_detected else 0.0 for g in trapped]),
        "corrected_answer_rate": _m([1.0 if g.answer_correct else 0.0 for g in trapped]),
        "wrong_undetected_rate": _m([1.0 if (not g.answer_correct and not g.trap_detected) else 0.0 for g in trapped]),
        "confident_wrong_rate": _m([1.0 if g.confident_wrong else 0.0 for g in trapped]),
        "crash_rate": _m([1.0 if g.detail.get("error") else 0.0 for g in (clean + trapped)]),
        "false_alarm_rate": _m([1.0 if g.false_alarm else 0.0 for g in clean]),
    }


def _pairs_by_family(grades):
    byf: dict[str, dict[tuple, dict]] = {}
    for g in grades:
        if g.paired:
            byf.setdefault(g.family, {}).setdefault(g.seed, {})[g.variant] = g.score
    out = {}
    for fam, seeds in byf.items():
        out[fam] = [(d["clean"], d["trapped"]) for d in seeds.values() if "clean" in d and "trapped" in d]
    return {f: ps for f, ps in out.items() if ps}


def _cluster_bootstrap(by_fam, statfn, B=2000, seed=0):
    fams = list(by_fam)
    if not fams:
        return None
    rng = random.Random(seed)
    nf = len(fams)
    vals = []
    for _ in range(B):
        sampled = [by_fam[fams[rng.randrange(nf)]] for _ in range(nf)]
        vals.append(statfn(sampled))
    vals.sort()
    return {"point": statfn(list(by_fam.values())),
            "ci95_low": vals[int(0.025 * B)], "ci95_high": vals[int(0.975 * B)],
            "n_families": nf, "n_pairs": sum(len(p) for p in by_fam.values())}


def _micro_gap(list_of_pairlists):
    flat = [p for ps in list_of_pairlists for p in ps]
    return mean(c - t for c, t in flat) if flat else 0.0


def _macro_gap(list_of_pairlists):
    per = [mean(c - t for c, t in ps) for ps in list_of_pairlists if ps]
    return mean(per) if per else 0.0


def _macro(grades, key):
    fams = sorted({g.family for g in grades})
    vals = [_block([g for g in grades if g.family == f])[key] for f in fams]
    vals = [v for v in vals if v is not None]
    return _m(vals)


@dataclass
class Scorecard:
    agent: str
    headline: dict
    macro: dict
    scenario_headline: dict
    overall_all_families: dict
    by_domain: dict
    by_family: dict
    by_trap: dict
    scenarios: dict
    n_grades: int
    failures: list = field(default_factory=list)

    def to_dict(self):
        return {k: getattr(self, k) for k in (
            "agent", "headline", "macro", "scenario_headline", "overall_all_families",
            "by_domain", "by_family", "by_trap", "scenarios", "n_grades", "failures")}

    def to_json(self, **kw):
        return json.dumps(self.to_dict(), indent=2, **kw)

    def to_markdown(self):
        def p(x):
            return "  -  " if x is None else f"{100 * x:.1f}"
        h, m = self.headline, self.macro
        ci, mci = h.get("gap_ci") or {}, m.get("gap_ci") or {}
        L = [f"# SciHarnessBench scorecard — `{self.agent}`", "",
             "_Headline over paired families; non-paired families are scenarios (below)._", "",
             f"- **Competence:** {p(h['competence'])}%   **Robustness:** {p(h['robustness'])}%",
             f"- **Fake-Science Gap:** {p(h['fake_science_gap'])} pts "
             f"(micro 95% CI [{p(ci.get('ci95_low'))}, {p(ci.get('ci95_high'))}]; "
             f"macro {p(m.get('fake_science_gap'))} pts CI [{p(mci.get('ci95_low'))}, {p(mci.get('ci95_high'))}])",
             f"- trapped breakdown — corrected: {p(h['corrected_answer_rate'])}%  "
             f"detected: {p(h['trap_detection_rate'])}%  wrong-undetected: {p(h['wrong_undetected_rate'])}%  "
             f"confident-wrong: {p(h['confident_wrong_rate'])}%  crash: {p(h['crash_rate'])}%",
             f"- **False-alarm (clean):** {p(h['false_alarm_rate'])}%",
             f"- scenario robustness (non-paired): {p(self.scenario_headline.get('robustness'))}%",
             f"- tasks graded: {self.n_grades}", "",
             "## By domain (paired)", "",
             "| domain | competence | robustness | gap | conf-wrong |", "|---|---:|---:|---:|---:|"]
        for d in sorted(self.by_domain):
            b = self.by_domain[d]
            L.append(f"| {d} | {p(b['competence'])} | {p(b['robustness'])} | "
                     f"{p(b['fake_science_gap'])} | {p(b['confident_wrong_rate'])} |")
        L += ["", "## By trap type (paired)", "",
              "| trap | robustness | detection | conf-wrong | n |", "|---|---:|---:|---:|---:|"]
        for t in sorted(self.by_trap):
            b = self.by_trap[t]
            L.append(f"| {t} | {p(b['robustness'])} | {p(b['trap_detection_rate'])} | "
                     f"{p(b['confident_wrong_rate'])} | {b['n_trapped']} |")
        if self.scenarios.get("families"):
            L += ["", "## Robustness scenarios (non-paired)", "",
                  "| family | competence | robustness |", "|---|---:|---:|"]
            for f, b in sorted(self.scenarios["families"].items()):
                L.append(f"| {f} | {p(b['competence'])} | {p(b['robustness'])} |")
        return "\n".join(L) + "\n"


def aggregate(agent_name: str, grades: list) -> Scorecard:
    paired = [g for g in grades if g.paired]
    nonpaired = [g for g in grades if not g.paired]

    headline = _block(paired)
    by_fam = _pairs_by_family(paired)
    headline["gap_ci"] = _cluster_bootstrap(by_fam, _micro_gap)
    macro = {"fake_science_gap": _macro(paired, "fake_science_gap"),
             "competence": _macro(paired, "competence"), "robustness": _macro(paired, "robustness"),
             "gap_ci": _cluster_bootstrap(by_fam, _macro_gap)}

    by_domain = {d: _block([g for g in paired if g.domain == d]) for d in sorted({g.domain for g in paired})}
    by_family = {f: _block([g for g in paired if g.family == f]) for f in sorted({g.family for g in paired})}
    by_trap = {t: _block([g for g in paired if g.trap_type == t]) for t in sorted({g.trap_type for g in paired if g.trap_type})}
    scenarios = {"overall": _block(nonpaired) if nonpaired else {},
                 "families": {f: _block([g for g in nonpaired if g.family == f])
                              for f in sorted({g.family for g in nonpaired})}}
    failures = [{"task_id": g.task_id, "variant": g.variant, "detail": g.detail}
                for g in grades if g.detail.get("error")]
    return Scorecard(agent_name, headline, macro, _block(nonpaired), _block(grades),
                     by_domain, by_family, by_trap, scenarios, len(grades), failures)
