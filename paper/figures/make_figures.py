#!/usr/bin/env python3
"""Publication-grade figures, generated from the benchmark's own JSON.

Every figure is data-driven and overlap-free: provider-colored horizontal bars
(no scatter-label collisions), a consistent serif style, vector PDF output.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(REPO, "results")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 200,
})

NAVY, RUST, TEAL, GOLD, GREY = "#27406b", "#b5462b", "#2a9d8f", "#d9a441", "#9aa3af"
PROVIDER = {"anthropic": "#b5462b", "openai": "#2a8f6b", "google": "#3b6fb0"}
SHORT = {
    "anthropic:claude-opus-4-8": "Claude Opus 4.8", "anthropic:claude-sonnet-4-6": "Claude Sonnet 4.6",
    "anthropic:claude-haiku-4-5-20251001": "Claude Haiku 4.5", "openai:gpt-5.5": "GPT-5.5",
    "openai:gpt-5.1": "GPT-5.1", "openai:gpt-4.1": "GPT-4.1", "openai:gpt-5-mini": "GPT-5 mini",
    "google:gemini-3.1-pro-preview": "Gemini 3.1 Pro", "google:gemini-2.5-pro": "Gemini 2.5 Pro",
    "google:gemini-2.5-flash": "Gemini 2.5 Flash",
}


def _load(name):
    return json.load(open(os.path.join(RESULTS, f"scorecard_reference-{name}.json")))


def _save(fig, name):
    fig.savefig(os.path.join(HERE, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote " + name)


def fig_discrimination():
    naive, careful = _load("naive"), _load("careful")
    groups = [("competence", "Competence\n(clean)"), ("robustness", "Robustness\n(trapped)"),
              ("trap_detection_rate", "Trap\ndetection")]
    nv = [100 * naive["headline"][k] for k, _ in groups]
    cv = [100 * careful["headline"][k] for k, _ in groups]
    x = range(len(groups))
    w = 0.36
    fig, ax = plt.subplots(figsize=(3.34, 2.15))
    b1 = ax.bar([i - w / 2 for i in x], nv, w, label="naive (trusts inputs)", color=RUST, edgecolor="white", linewidth=0.4)
    b2 = ax.bar([i + w / 2 for i in x], cv, w, label="careful (validates)", color=NAVY, edgecolor="white", linewidth=0.4)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.6, f"{b.get_height():.0f}",
                    ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([lab for _, lab in groups])
    ax.set_ylabel("score (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, handlelength=1.2)
    _save(fig, "fig_discrimination.pdf")


def _models():
    path = os.path.join(RESULTS, "models", "_summary.json")
    return json.load(open(path)).get("models", {}) if os.path.exists(path) else {}


def fig_models():
    """Competence vs robustness per model, horizontal bars (no label overlap)."""
    models = _models()
    if not models:
        print("(no model results; skipping fig_models.pdf)")
        return
    order = sorted(models, key=lambda k: models[k]["robustness"])
    names = [SHORT.get(s, s) for s in order]
    comp = [100 * models[s]["competence"] for s in order]
    rob = [100 * models[s]["robustness"] for s in order]
    y = range(len(order))
    h = 0.38
    fig, ax = plt.subplots(figsize=(3.34, 3.5))
    ax.barh([i + h / 2 for i in y], comp, h, label="competence (clean)", color=GREY)
    ax.barh([i - h / 2 for i in y], rob, h, label="robustness (trapped)", color=NAVY)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.set_xlabel("score (%)")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, loc="lower right", handlelength=1.1)
    ax.tick_params(axis="y", length=0)
    _save(fig, "fig_models.pdf")


def fig_failuremodes():
    """The two failure modes per model: confident-wrong vs false-alarm."""
    models = _models()
    if not models:
        print("(no model results; skipping fig_failuremodes.pdf)")
        return
    order = sorted(models, key=lambda k: -models[k]["confident_wrong_rate"])
    names = [SHORT.get(s, s) for s in order]
    cw = [100 * models[s]["confident_wrong_rate"] for s in order]
    fa = [100 * models[s]["false_alarm_rate"] for s in order]
    y = range(len(order))
    h = 0.38
    fig, ax = plt.subplots(figsize=(3.34, 3.5))
    ax.barh([i + h / 2 for i in y], cw, h, label="confident-wrong (fake science)", color=RUST)
    ax.barh([i - h / 2 for i in y], fa, h, label="false-alarm (over-flagging)", color=GOLD)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.set_xlabel("rate (%)")
    ax.set_xlim(0, 46)
    ax.legend(frameon=False, loc="upper right", handlelength=1.1)
    ax.tick_params(axis="y", length=0)
    _save(fig, "fig_failuremodes.pdf")


if __name__ == "__main__":
    fig_discrimination()
    fig_models()
    fig_failuremodes()
