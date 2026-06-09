"""Run the SciHarnessBench model evaluation on Modal, fanned out in parallel.

One Modal task per (model, family); results are aggregated locally into per-model
scorecards under results/models/. Keys are injected as a Modal secret built from
the local key files (never committed). Usage:

    # validate one (model, family)
    modal run scripts/modal_eval.py --n-seeds 1 --models openai:gpt-4.1 --families chem.reaction_energy
    # full parallel run (all models, all families)
    modal run scripts/modal_eval.py --n-seeds 5 --prompt-style uncued
"""
import json
import os
import sys

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _key(name):
    return open(os.path.expanduser(f"~/.shb_{name}_key")).read().strip()


SECRET = modal.Secret.from_dict({
    "OPENAI_API_KEY": _key("openai"),
    "ANTHROPIC_API_KEY": _key("anthropic"),
    "GOOGLE_API_KEY": _key("google"),
})

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libxrender1", "libxext6", "libsm6", "libgomp1")          # rdkit runtime libs
    .pip_install("numpy", "scipy", "scikit-learn", "pandas", "rdkit", "biopython",
                 "astropy", "openai>=2", "anthropic>=0.40", "google-genai")
    .add_local_dir(os.path.join(REPO, "shb"), remote_path="/root/shb")
)
app = modal.App("shb-eval")

MODELS = [
    "anthropic:claude-opus-4-8", "anthropic:claude-sonnet-4-6", "anthropic:claude-haiku-4-5-20251001",
    "openai:gpt-5.5", "openai:gpt-5.1", "openai:gpt-4.1", "openai:gpt-5-mini",
    "google:gemini-3.1-pro-preview", "google:gemini-2.5-pro", "google:gemini-2.5-flash",
]


@app.function(image=image, secrets=[SECRET], timeout=3000, retries=1, max_containers=24)
def eval_one(spec, family_id, n_seeds, prompt_style):
    sys.path.insert(0, "/root")
    from shb import run_benchmark
    from shb.providers import build_agent
    grades = run_benchmark(build_agent(spec), seeds=range(n_seeds),
                           families=[family_id], prompt_style=prompt_style)
    return spec, grades


@app.local_entrypoint()
def main(n_seeds: int = 5, prompt_style: str = "uncued", models: str = "", families: str = ""):
    sys.path.insert(0, REPO)
    from shb import aggregate, registry
    registry.ensure_loaded()
    specs = models.split(",") if models else MODELS
    fams = families.split(",") if families else [f.family_id for f in registry.all_families()]
    args = [(s, fam, n_seeds, prompt_style) for s in specs for fam in fams]
    print(f"[modal] dispatching {len(args)} (model x family) jobs "
          f"({len(specs)} models x {len(fams)} families, {n_seeds} seeds, {prompt_style})...")

    by_spec = {}
    for spec, grades in eval_one.starmap(args, order_outputs=False):
        by_spec.setdefault(spec, []).extend(grades)

    out = os.path.join(REPO, "results", "models")
    os.makedirs(out, exist_ok=True)
    summary = {}
    for spec in sorted(by_spec):
        card = aggregate(spec, by_spec[spec])
        with open(os.path.join(out, spec.replace(":", "__").replace("/", "_") + ".json"), "w") as fh:
            fh.write(card.to_json())
        h = card.headline
        summary[spec] = h
        print(f"{spec:40s} C={100*h['competence']:5.1f} R={100*h['robustness']:5.1f} "
              f"gap={100*h['fake_science_gap']:5.1f} conf-wrong={100*h['confident_wrong_rate']:5.1f} "
              f"false-alarm={100*h['false_alarm_rate']:5.1f}")
    json.dump({"n_seeds": n_seeds, "prompt_style": prompt_style, "models": summary},
              open(os.path.join(out, "_summary.json"), "w"), indent=2)
    print(f"\n[modal] wrote per-model scorecards to results/models/ ({len(summary)} models)")
