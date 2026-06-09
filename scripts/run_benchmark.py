#!/usr/bin/env python3
"""Run an agent against SciHarnessBench and emit a scorecard.

Examples
--------
    # the built-in reference agents (no API key needed)
    python scripts/run_benchmark.py --agent reference-careful --seeds 5
    python scripts/run_benchmark.py --agent reference-naive   --seeds 5

    # both, with a side-by-side summary (this is the discrimination demo)
    python scripts/run_benchmark.py --agent both --seeds 5 --domains chemistry statistics

To test a real model, write a ``complete(prompt)->str`` callable and wrap it in
``shb.agents.LLMAgent`` (see shb/agents/llm.py), then pass it to run_benchmark().
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shb import aggregate, run_benchmark  # noqa: E402
from shb.agents import ReferenceAgent  # noqa: E402


def _summary_line(name, card) -> str:
    o = card.headline

    def p(x):
        return "  - " if x is None else f"{100 * x:5.1f}%"
    return (f"{name:20s} competence={p(o['competence'])}  robustness={p(o['robustness'])}  "
            f"gap={p(o['fake_science_gap'])}  confident-wrong={p(o['confident_wrong_rate'])}  "
            f"false-alarm={p(o['false_alarm_rate'])}  trap-detect={p(o['trap_detection_rate'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="both",
                    choices=["both", "reference-naive", "reference-careful"])
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds per family")
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="write the (last) scorecard markdown here")
    ap.add_argument("--json", default=None, help="write the (last) scorecard JSON here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    names = ["reference-naive", "reference-careful"] if args.agent == "both" else [args.agent]
    seeds = range(args.seeds)
    cards = {}
    for name in names:
        agent = ReferenceAgent(name.split("-", 1)[1])
        grades = run_benchmark(agent, seeds=seeds, domains=args.domains,
                               families=args.families, progress=not args.quiet)
        cards[name] = aggregate(name, grades)

    print("\n=== SciHarnessBench summary ===")
    for name in names:
        print(_summary_line(name, cards[name]))
    if {"reference-naive", "reference-careful"} <= set(cards):
        gap_n = cards["reference-naive"].headline["fake_science_gap"]
        gap_c = cards["reference-careful"].headline["fake_science_gap"]
        print(f"\nDiscrimination: naive gap {100*gap_n:.1f} pts vs careful gap {100*gap_c:.1f} pts "
              f"(a working benchmark shows naive >> careful).")

    last = cards[names[-1]]
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(last.to_markdown())
        print(f"\nwrote {args.out}")
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(last.to_json())
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
