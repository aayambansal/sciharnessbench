#!/usr/bin/env python3
"""Generate the paper's figures directly from the benchmark scorecard JSON.

Every figure is data-driven: rerun after any benchmark change and the paper stays
in sync. Outputs vector PDFs into this directory.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(REPO, "results")

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
NAVY, RUST = "#28406b", "#b5482a"


def load(agent):
    return json.load(open(os.path.join(RESULTS, f"scorecard_reference-{agent}.json")))


def fig_discrimination():
    naive, careful = load("naive"), load("careful")
    metrics = [("competence", "Competence\n(clean)"),
               ("robustness", "Robustness\n(trapped)"),
               ("trap_detection_rate", "Trap\ndetection")]
    nv = [100 * naive["headline"][k] for k, _ in metrics]
    cv = [100 * careful["headline"][k] for k, _ in metrics]
    x = range(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    b1 = ax.bar([i - w / 2 for i in x], nv, w, label="naive (trusts inputs)", color=RUST)
    b2 = ax.bar([i + w / 2 for i in x], cv, w, label="careful (validates)", color=NAVY)
    ax.set_xticks(list(x))
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("score (\\%)")
    ax.set_ylim(0, 108)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=6.5)
    ax.legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(0.0, 0.55))
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_discrimination.pdf"))
    plt.close(fig)


def fig_by_domain():
    naive, careful = load("naive"), load("careful")
    doms = sorted(naive["by_domain"])
    gap = [100 * naive["by_domain"][d]["fake_science_gap"] for d in doms]
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.barh(doms, gap, color=NAVY, height=0.62)
    ax.set_xlabel("fake-science gap (pts), naive agent")
    ax.set_xlim(0, 105)
    for i, g in enumerate(gap):
        ax.text(g + 1, i, f"{g:.0f}", va="center", fontsize=6.5)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_bydomain.pdf"))
    plt.close(fig)


SHORT = {
    "anthropic:claude-opus-4-8": "Opus 4.8", "anthropic:claude-sonnet-4-6": "Sonnet 4.6",
    "anthropic:claude-haiku-4-5-20251001": "Haiku 4.5", "openai:gpt-5.5": "GPT-5.5",
    "openai:gpt-5.1": "GPT-5.1", "openai:gpt-4.1": "GPT-4.1", "openai:gpt-5-mini": "GPT-5 mini",
    "google:gemini-3.1-pro-preview": "Gemini 3.1 Pro", "google:gemini-2.5-pro": "Gemini 2.5 Pro",
    "google:gemini-2.5-flash": "Gemini 2.5 Flash",
}


def fig_models():
    """Competence vs robustness per model; distance below the diagonal is the gap."""
    path = os.path.join(RESULTS, "models", "_summary.json")
    if not os.path.exists(path):
        print("(no model results yet; skipping fig_models.pdf)")
        return
    models = json.load(open(path)).get("models", {})
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.plot([0, 100], [0, 100], ls="--", lw=0.7, color="#999", zorder=0)
    ax.text(62, 70, "no fake science", rotation=45, fontsize=6, color="#999", ha="center", va="center")
    for spec, h in models.items():
        c, r = 100 * h["competence"], 100 * h["robustness"]
        ax.scatter(c, r, s=22, color=NAVY, zorder=3)
        ax.annotate(SHORT.get(spec, spec.split(":")[-1]), (c, r), fontsize=5.5,
                    xytext=(3, -1), textcoords="offset points")
    ax.set_xlabel("competence (clean, \\%)")
    ax.set_ylabel("robustness (trapped, \\%)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_models.pdf"))
    plt.close(fig)
    print("wrote fig_models.pdf")


if __name__ == "__main__":
    fig_discrimination()
    fig_by_domain()
    fig_models()
    print("wrote fig_discrimination.pdf, fig_bydomain.pdf")
