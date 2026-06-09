#!/usr/bin/env python3
"""Generate the reference scorecard artifact under results/.

Runs both built-in reference agents over the whole suite and writes a combined
markdown report plus per-agent JSON. The two agents bracket the achievable
range: naive (trusts every input) is the fake-science floor; careful (validates
everything) is the ceiling. A real model under test lands between them.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shb import aggregate, registry, run_benchmark  # noqa: E402
from shb.agents import ReferenceAgent  # noqa: E402

SEEDS = int(os.environ.get("SHB_SEEDS", "12"))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def pct(x):
    return "  -  " if x is None else f"{100 * x:.1f}"


def combined_markdown(naive, careful, n_seeds, n_grades) -> str:
    hn, hc = naive.headline, careful.headline
    ci = hn.get("gap_ci") or {}
    fams = registry.all_families()
    npaired = sum(1 for f in fams if f.paired)
    L = [
        "# SciHarnessBench — reference scorecard", "",
        "SciHarnessBench measures **fake science**: does a scientific agent produce a confident,",
        "well-formatted, and *wrong* result when the data hides a flaw a competent scientist would",
        "catch? Every task ships as a clean/trapped pair; the headline metric is the **gap** between",
        "an agent's accuracy on the two, computed over *paired* families (true counterfactual twins).",
        "",
        "The two reference agents use **no model and no API key** — they bound the range a real",
        "system under test falls into. `reference-naive` does the right computation but trusts every",
        "input; `reference-careful` validates inputs and reports structured, evidence-bearing flaws.",
        "",
        f"_{n_seeds} seeds x {len(fams)} families ({npaired} paired, {len(fams)-npaired} robustness "
        f"scenarios) = {n_grades} graded tasks per agent. Self-generated and self-graded._", "",
        "## Headline (paired families)", "",
        "| agent | competence | robustness | fake-science gap | confident-wrong | false-alarm | trap detection |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| `reference-naive` | {pct(hn['competence'])}% | {pct(hn['robustness'])}% | "
        f"**{pct(hn['fake_science_gap'])} pts** | {pct(hn['confident_wrong_rate'])}% | "
        f"{pct(hn['false_alarm_rate'])}% | {pct(hn['trap_detection_rate'])}% |",
        f"| `reference-careful` | {pct(hc['competence'])}% | {pct(hc['robustness'])}% | "
        f"**{pct(hc['fake_science_gap'])} pts** | {pct(hc['confident_wrong_rate'])}% | "
        f"{pct(hc['false_alarm_rate'])}% | {pct(hc['trap_detection_rate'])}% |",
        "",
        f"Naive fake-science gap 95% bootstrap CI: [{pct(ci.get('ci95_low'))}, {pct(ci.get('ci95_high'))}] pts "
        f"over {ci.get('n_pairs', 0)} pairs. Both agents are equally competent; they diverge entirely on",
        "robustness. The gap is a within-task difference (robust to grading noise) and cannot be gamed",
        "by abstaining (a flag/abstention on a clean task is a false alarm that scores zero).",
        "",
        "## Robustness by domain (paired)", "",
        "| domain | competence | naive robustness | careful robustness |", "|---|---:|---:|---:|",
    ]
    for d in sorted(careful.by_domain):
        L.append(f"| {d} | {pct(careful.by_domain[d]['competence'])} "
                 f"| {pct(naive.by_domain[d]['robustness'])} | {pct(careful.by_domain[d]['robustness'])} |")
    L += ["", "## Robustness by trap type (paired)", "",
          "| trap type | naive robustness | careful robustness | careful trap detection | n |",
          "|---|---:|---:|---:|---:|"]
    for t in sorted(careful.by_trap):
        L.append(f"| {t} | {pct(naive.by_trap[t]['robustness'])} | {pct(careful.by_trap[t]['robustness'])} "
                 f"| {pct(careful.by_trap[t]['trap_detection_rate'])} | {careful.by_trap[t]['n_trapped']} |")
    if careful.scenarios.get("families"):
        L += ["", "## Robustness scenarios (non-paired families)", "",
              "| family | naive robustness | careful robustness |", "|---|---:|---:|"]
        for f in sorted(careful.scenarios["families"]):
            L.append(f"| {f} | {pct(naive.scenarios['families'][f]['robustness'])} "
                     f"| {pct(careful.scenarios['families'][f]['robustness'])} |")
    L += ["", f"## Task families ({len(fams)})", ""]
    cur = None
    for f in fams:
        if f.domain != cur:
            cur = f.domain
            L.append(f"\n**{cur}**")
        tag = "" if f.paired else " _(scenario)_"
        L.append(f"- `{f.family_id}` — {f.title} _(trap: {f.flaw_kind})_{tag}")
    return "\n".join(L) + "\n"


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    cards = {}
    for mode in ("naive", "careful"):
        grades = run_benchmark(ReferenceAgent(mode), seeds=range(SEEDS))
        cards[mode] = aggregate(f"reference-{mode}", grades)
        with open(os.path.join(OUT, f"scorecard_reference-{mode}.json"), "w") as fh:
            fh.write(cards[mode].to_json())
    md = combined_markdown(cards["naive"], cards["careful"], SEEDS, cards["careful"].n_grades)
    with open(os.path.join(OUT, "sample_scorecard.md"), "w") as fh:
        fh.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
