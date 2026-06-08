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

SEEDS = int(os.environ.get("SHB_SEEDS", "10"))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def pct(x):
    return "  -  " if x is None else f"{100 * x:.1f}"


def combined_markdown(naive, careful, n_seeds, n_families, n_grades) -> str:
    no, co = naive.overall, careful.overall
    L = [
        "# SciHarnessBench — reference scorecard",
        "",
        "SciHarnessBench measures **fake science**: does a scientific agent produce a confident,",
        "well-formatted, and *wrong* result when the data hides a flaw a competent scientist would",
        "catch? Every task ships as a clean/trapped pair; the headline metric is the **gap** between",
        "an agent's accuracy on the two.",
        "",
        "The two reference agents below use **no model and no API key** — they bound the range a real",
        "system under test falls into:",
        "",
        "- `reference-naive` does the right computation but trusts every input at face value → commits the fake science.",
        "- `reference-careful` validates inputs, checks units/convergence/confounds, and corrects or flags the flaw → does real science.",
        "",
        f"_Generated over {n_seeds} seeds × {n_families} families = {n_grades} graded tasks "
        f"(clean + trapped), fully self-generated and self-graded._",
        "",
        "## Headline",
        "",
        "| agent | competence (clean) | robustness (trapped) | fake-science gap | trap detection |",
        "|---|---:|---:|---:|---:|",
        f"| `reference-naive` | {pct(no['competence'])}% | {pct(no['robustness'])}% | "
        f"**{pct(no['fake_science_gap'])} pts** | {pct(no['trap_detection_rate'])}% |",
        f"| `reference-careful` | {pct(co['competence'])}% | {pct(co['robustness'])}% | "
        f"**{pct(co['fake_science_gap'])} pts** | {pct(co['trap_detection_rate'])}% |",
        "",
        "Both agents are equally **competent** (they solve clean tasks). They diverge entirely on",
        "**robustness**: the naive agent collapses under planted flaws while the careful agent holds.",
        "The gap is the product — and because it is a within-task difference, it is robust to grading",
        "noise and cannot be gamed by reflexive abstention (abstaining on a clean task scores zero).",
        "",
        "## Robustness by domain (trapped pass rate)",
        "",
        "| domain | naive | careful |",
        "|---|---:|---:|",
    ]
    for dom in sorted(careful.by_domain):
        L.append(f"| {dom} | {pct(naive.by_domain[dom]['robustness'])}% "
                 f"| {pct(careful.by_domain[dom]['robustness'])}% |")
    L += ["", "## Robustness by trap type (trapped pass rate)", "",
          "| trap type | naive | careful | n |", "|---|---:|---:|---:|"]
    for trap in sorted(careful.by_trap):
        b_n, b_c = naive.by_trap[trap], careful.by_trap[trap]
        L.append(f"| {trap} | {pct(b_n['robustness'])}% | {pct(b_c['robustness'])}% | {b_c['n_trapped']} |")

    fams = registry.all_families()
    L += ["", f"## Task families ({len(fams)})", ""]
    cur = None
    for f in fams:
        if f.domain != cur:
            cur = f.domain
            L.append(f"\n**{cur}**")
        traps = ", ".join(t.value for t in f.trap_types)
        L.append(f"- `{f.family_id}` — {f.title} _(traps: {traps})_")
    return "\n".join(L) + "\n"


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    cards = {}
    for mode in ("naive", "careful"):
        grades = run_benchmark(ReferenceAgent(mode), seeds=range(SEEDS), progress=False)
        cards[mode] = aggregate(f"reference-{mode}", grades)
        with open(os.path.join(OUT, f"scorecard_reference-{mode}.json"), "w") as fh:
            fh.write(cards[mode].to_json())

    n_fam = len(registry.all_families())
    n_grades = cards["careful"].n_grades
    md = combined_markdown(cards["naive"], cards["careful"], SEEDS, n_fam, n_grades)
    with open(os.path.join(OUT, "sample_scorecard.md"), "w") as fh:
        fh.write(md)

    print(md)
    print(f"\nwrote results/sample_scorecard.md and per-agent JSON ({n_fam} families, {n_grades} tasks/agent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
