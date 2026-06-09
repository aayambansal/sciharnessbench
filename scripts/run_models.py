"""Local parallel model evaluation: every (model, family, seed, variant) graded
concurrently via a thread pool (the work is API-bound). Writes per-model
scorecards to results/models/ and a _summary.json. Reads keys from the
environment (OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY).

    python scripts/run_models.py --seeds 5 --prompt-style uncued --workers 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shb import aggregate, registry, run_instance  # noqa: E402
from shb.providers import build_agent  # noqa: E402

MODELS = [
    "anthropic:claude-opus-4-8", "anthropic:claude-sonnet-4-6", "anthropic:claude-haiku-4-5-20251001",
    "openai:gpt-5.5", "openai:gpt-5.1", "openai:gpt-4.1", "openai:gpt-5-mini",
    "google:gemini-3.1-pro-preview", "google:gemini-2.5-pro", "google:gemini-2.5-flash",
]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--prompt-style", default="uncued", choices=["cued", "uncued"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--models", default="")
    args = ap.parse_args()

    specs = args.models.split(",") if args.models else MODELS
    agents = {s: build_agent(s) for s in specs}
    fams = registry.all_families()
    root = tempfile.mkdtemp(prefix="shb_models_")
    jobs = [(s, fam, seed, v) for s in specs for fam in fams
            for seed in range(args.seeds) for v in fam.variants]
    grades = defaultdict(list)
    counts = defaultdict(int)
    lock = threading.Lock()
    total = len(jobs)
    print(f"[run_models] {len(specs)} models x {len(fams)} families x {args.seeds} seeds "
          f"= {total} tasks ({args.prompt_style}), {args.workers} workers", flush=True)

    def work(job):
        spec, fam, seed, variant = job
        inst = fam.generate(seed, variant)
        return spec, run_instance(agents[spec], fam, inst, root, args.prompt_style)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, j) for j in jobs]):
            try:
                spec, g = fut.result()
                with lock:
                    grades[spec].append(g)
                    counts[spec] += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[run_models] job error: {exc!r}", flush=True)
            done += 1
            if done % 100 == 0 or done == total:
                print(f"[run_models] {done}/{total} done", flush=True)

    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for spec in sorted(grades):
        card = aggregate(spec, grades[spec])
        with open(os.path.join(OUT, spec.replace(":", "__").replace("/", "_") + ".json"), "w") as fh:
            fh.write(card.to_json())
        h = card.headline
        summary[spec] = h
        print(f"[result] {spec:40s} C={100*h['competence']:5.1f} R={100*h['robustness']:5.1f} "
              f"gap={100*h['fake_science_gap']:5.1f} conf-wrong={100*h['confident_wrong_rate']:5.1f} "
              f"false-alarm={100*h['false_alarm_rate']:5.1f}", flush=True)
    json.dump({"n_seeds": args.seeds, "prompt_style": args.prompt_style, "models": summary},
              open(os.path.join(OUT, "_summary.json"), "w"), indent=2)
    print(f"[run_models] DONE — wrote {len(summary)} scorecards to results/models/", flush=True)


if __name__ == "__main__":
    main()
