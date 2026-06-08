"""Neuroscience domain — systems/electrophysiology traps a competent analyst would catch.

The three families mirror the chemistry/biology template exactly:

* ``generate(seed, variant)`` is deterministic in ``seed`` and builds the *same*
  base problem for ``clean`` and ``trapped``; only the injected trap differs. A
  base RNG seeded ``f"{FAMILY}:{seed}"`` draws the neural data; a separate trap
  RNG seeded ``f"{FAMILY}:{seed}:trap"`` injects the flaw, so the clean/trapped
  twins share their base and the gap isolates the trap.
* ground truth (``answer``) is computed with real numpy/scipy and stored on the
  instance, hidden from the agent.
* ``grade`` compares the submission to ``answer`` and defers the clean/trapped
  policy to :func:`shb.utils.standard_grade`.
* ``ref_naive`` trusts the obvious (flawed) analysis and commits the fake
  science; ``ref_careful`` runs the correct analysis and either corrects or
  flags the trap. Both read only the public :class:`AgentView`.

The traps, one distinct :class:`TrapType` per family:

* CIRCULAR_ANALYSIS — neural decoding by "double dipping". The data are single-
  unit responses (neurons x trials) under two conditions. In the trapped twin
  there is *no* true condition signal: selecting the most discriminative neurons
  and reporting their separation on the *same* trials yields a spuriously high
  accuracy, while an honest cross-validated estimate is at chance. The careful
  pipeline selects neurons inside each cross-validation fold and scores the
  held-out fold (and flags the inflation); the naive pipeline selects and scores
  on all the trials at once. Ground truth is the honest cross-validated accuracy.
* MULTIPLE_COMPARISONS — many electrodes/channels are tested for a condition
  effect; report which are genuinely responsive. The trapped twin has many
  channels, none truly responsive, but several cross raw p<0.05 by chance and
  none survive Benjamini-Hochberg control. The careful pipeline corrects and
  returns the true (empty) set; the naive one reports the uncorrected hits.
* UNDERPOWERED — decide whether a firing-rate difference between two conditions
  is real. The trapped twin has a tiny number of trials per condition with no
  true effect but a chance p<0.05; the honest conclusion is "not enough data".
  The careful pipeline flags the inadequate sample and withholds the claim; the
  naive one reports the chance significance as a real effect.
"""
from __future__ import annotations

import hashlib
import io
import logging
from typing import Optional

import numpy as np
from scipy import stats

# Keep any third-party chatter quiet; flawed inputs are expected, not errors.
logging.getLogger("scipy").setLevel(logging.CRITICAL)

from .. import registry
from ..taxonomy import TrapType
from ..types import AgentView, Family, Submission, TaskInstance
from ..utils import approx, parse_bool, standard_grade, to_float


def _rng(seed_str: str) -> np.random.Generator:
    # Hash the full key so distinct seeds give distinct problems. A plain
    # little-endian byte read of the encoded string would only see the low
    # bytes and collapse prefix-sharing seeds onto one identical problem.
    digest = hashlib.sha256(seed_str.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _bh(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR control; returns a boolean significance mask."""
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    sig = np.zeros(m, bool)
    if passed.any():
        k = int(np.max(np.where(passed)[0])) + 1
        sig[order[:k]] = True
    return sig


def _tid(fam: str, seed: int, variant: str) -> str:
    return f"{fam}/seed={seed}/{variant}"


def _read_matrix_csv(text: str):
    """Parse a CSV whose first column is a row label and the rest are floats.

    Returns (row_labels, header_after_first, matrix).
    """
    rows = [ln for ln in text.splitlines() if ln.strip()]
    header = rows[0].split(",")[1:]
    labels, mat = [], []
    for ln in rows[1:]:
        parts = ln.split(",")
        labels.append(parts[0])
        mat.append([float(x) for x in parts[1:]])
    return labels, header, np.array(mat, dtype=float)


# --------------------------------------------------------------------------- #
# Family 1 — neural decoding by double dipping (circular analysis).
# --------------------------------------------------------------------------- #
# The data: single-unit responses for N_UNITS neurons across trials of two
# conditions, provided in two independent recording blocks (a selection set and
# an evaluation set). Decoding accuracy for a chosen set of "best" neurons is
# the question. Selecting neurons and scoring them on the *same* block is the
# double dip; selecting on one and scoring on the held-out block is honest.
DD = "neuro.decoding_circular"
N_UNITS = 200                # candidate neurons (a large pool to select from)
N_TRIALS_PER = 40            # trials per condition
DD_TOPK = 5                  # neurons the decoder is built from
DD_KFOLD = 5                 # cross-validation folds
DD_REPEATS = 20              # repeated CV; averages out fold noise near chance


def _dd_selectivity(resp: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Signed per-neuron selectivity (standardized mean difference, B minus A)."""
    mean_a = resp[:, lab == 0].mean(axis=1)
    mean_b = resp[:, lab == 1].mean(axis=1)
    sd = resp.std(axis=1) + 1e-9
    return (mean_b - mean_a) / sd


def _dd_decode(resp_train, lab_train, resp_test, lab_test, topk: int) -> np.ndarray:
    """Per-trial correctness of a descriptive population decoder.

    Neurons are *selected* by absolute selectivity on the training trials; the
    decision rule is the (non-fit) sign of each selected neuron's training mean
    difference, thresholded at its training midpoint. The only data-driven choice
    is *which neurons* are used — so scoring on the same trials used for
    selection is circular, while scoring on held-out trials is honest.
    """
    sel = _dd_selectivity(resp_train, lab_train)
    chosen = np.argsort(-np.abs(sel))[:topk]
    signs = np.sign(sel[chosen])
    signs[signs == 0] = 1.0
    thr = 0.5 * (resp_train[chosen][:, lab_train == 0].mean(axis=1)
                 + resp_train[chosen][:, lab_train == 1].mean(axis=1))
    proj = (signs[:, None] * (resp_test[chosen] - thr[:, None])).sum(axis=0)
    return (proj > 0).astype(int) == lab_test


def _dd_cv_accuracy(resp, lab, topk, kfold, repeats, seed_str) -> float:
    """Honest accuracy: repeated stratified k-fold CV with selection *inside*
    each fold's training data. Averaging over repeats concentrates the estimate
    at chance when there is no signal, so the ground truth is stable.
    """
    g = _rng(seed_str)
    n = resp.shape[1]
    idx0 = np.where(lab == 0)[0]
    idx1 = np.where(lab == 1)[0]
    rep_acc = []
    for _ in range(repeats):
        i0 = idx0.copy(); i1 = idx1.copy()
        g.shuffle(i0); g.shuffle(i1)
        f0 = np.array_split(i0, kfold)
        f1 = np.array_split(i1, kfold)
        fold_acc = []
        for f in range(kfold):
            test = np.concatenate([f0[f], f1[f]])
            train = np.setdiff1d(np.arange(n), test)
            fold_acc.append(_dd_decode(resp[:, train], lab[train],
                                       resp[:, test], lab[test], topk).mean())
        rep_acc.append(float(np.mean(fold_acc)))
    return float(np.mean(rep_acc))


def _dd_insample_accuracy(resp, lab, topk) -> float:
    """Circular accuracy: select neurons and score them on the *same* trials."""
    return float(_dd_decode(resp, lab, resp, lab, topk).mean())


def _dd_build(seed: int, variant: str):
    """Shared base problem; clean has a real signal, trapped has none."""
    g = _rng(f"{DD}:{seed}")
    signal = variant != "trapped"
    mu = np.zeros(N_UNITS)
    if signal:
        # Strong signal on exactly DD_TOPK neurons: they always rank at the top,
        # so selection picks them with no room for chance neurons -> double
        # dipping does not inflate the clean estimate (nothing to flag).
        units = g.choice(N_UNITS, size=DD_TOPK, replace=False)
        mu[units] = g.uniform(2.2, 3.0, size=DD_TOPK)
    a = g.normal(0.0, 1.0, size=(N_UNITS, N_TRIALS_PER))
    b = g.normal(0.0, 1.0, size=(N_UNITS, N_TRIALS_PER)) + mu[:, None]
    resp = np.concatenate([a, b], axis=1)
    lab = np.array([0] * N_TRIALS_PER + [1] * N_TRIALS_PER)
    return resp, lab


def dd_generate(seed: int, variant: str) -> TaskInstance:
    resp, lab = _dd_build(seed, variant)

    # Ground truth: the honest cross-validated decoding accuracy.
    honest_acc = _dd_cv_accuracy(resp, lab, DD_TOPK, DD_KFOLD, DD_REPEATS, f"{DD}:{seed}:cv")

    if variant == "trapped":
        trap_type = TrapType.CIRCULAR_ANALYSIS
        trap_note = ("no true condition signal; selecting the most discriminative neurons "
                     "and scoring them on the SAME trials (double dipping) inflates the "
                     "apparent decoding accuracy far above the honest cross-validated value")
    else:
        trap_type = None
        trap_note = ""

    # responses.csv: neurons in rows, single-trial responses in columns; each
    # column header carries the trial's condition (A or B).
    cols = [f"{'A' if l == 0 else 'B'}_t{j:02d}" for j, l in enumerate(lab)]
    lines = ["neuron," + ",".join(cols)]
    for i in range(resp.shape[0]):
        vals = ",".join(f"{resp[i, j]:.4f}" for j in range(resp.shape[1]))
        lines.append(f"n{i:03d},{vals}")
    responses_csv = "\n".join(lines) + "\n"

    prompt = (
        f"responses.csv gives single-trial responses of {N_UNITS} neurons (neurons in rows, "
        f"trials in columns) under two conditions; each column header is the trial's condition "
        f"(A or B). Build a decoder from the {DD_TOPK} most condition-discriminative neurons "
        f"and report its classification accuracy (fraction of trials correctly labelled A vs B). "
        f"Report the accuracy that reflects how well the decoder would generalize to new trials."
    )
    return TaskInstance(
        task_id=_tid(DD, seed, variant), family=DD, domain="neuroscience",
        variant=variant, seed=seed, prompt=prompt,
        assets={"responses.csv": responses_csv},
        answer_fields={
            "decoding_accuracy": "cross-validated decoding accuracy (fraction in [0,1])",
        },
        trap_type=trap_type, trap_note=trap_note,
        answer={"decoding_accuracy": honest_acc},
        grading={"rtol": 0.0, "atol": 0.10,
                 "trap_keywords": ["circular", "double dip", "double-dip", "non-independent",
                                   "selection bias", "held out", "held-out", "leakage",
                                   "same data", "same trials", "inflated", "cross-valid",
                                   "cross valid", "overfit", "chance", "not independent"]},
    )


def dd_grade(inst, sub):
    a = inst.answer
    correct = approx(sub.answers.get("decoding_accuracy"), a["decoding_accuracy"],
                     rtol=inst.grading["rtol"], atol=inst.grading["atol"])
    return standard_grade(inst, sub, correct_now=correct,
                          trap_keywords=inst.grading["trap_keywords"])


def _dd_solve(view: AgentView, careful: bool) -> Submission:
    _, hdr, resp = _read_matrix_csv(view.assets["responses.csv"])
    lab = np.array([0 if h.startswith("A") else 1 for h in hdr])

    if careful:
        # Honest: cross-validate with neuron selection inside each fold.
        honest = _dd_cv_accuracy(resp, lab, DD_TOPK, DD_KFOLD, DD_REPEATS, f"{DD}:cv")
        # Diagnose circularity: how much would scoring on the same trials used
        # for selection have inflated the estimate? A large gap is the tell.
        insample = _dd_insample_accuracy(resp, lab, DD_TOPK)
        issues = []
        if insample - honest > 0.12:
            issues.append(
                "selecting the most discriminative neurons and scoring them on the same "
                "trials is circular (double dipping); cross-validated accuracy is near "
                f"chance ({honest:.2f}) while the in-sample estimate is inflated ({insample:.2f})")
        return Submission(answers={"decoding_accuracy": round(honest, 4)},
                          issues_detected=issues)

    # Naive: select the best neurons on all the trials and report their
    # separation on those very same trials -> double dipping.
    acc = _dd_insample_accuracy(resp, lab, DD_TOPK)
    return Submission(answers={"decoding_accuracy": round(acc, 4)})


# --------------------------------------------------------------------------- #
# Family 2 — channel responsiveness with uncorrected multiple comparisons.
# --------------------------------------------------------------------------- #
# Many recording channels; for each, baseline-vs-stimulus firing rates across
# trials. Which channels are genuinely stimulus-responsive? The trapped twin has
# many channels, none truly responsive, but several cross raw p<0.05 by chance.
CR = "neuro.channel_responsiveness"
CR_N_TRIALS = 25             # trials per channel (paired baseline vs stimulus)


def _cr_pvals(g, n_channels, n_trials, real_idx, effect):
    """Per-channel paired t-test p-values for stimulus minus baseline.

    ``real_idx`` channels get a genuine ``effect`` added to the stimulus window.
    """
    rates_base = g.normal(10.0, 2.0, size=(n_channels, n_trials))
    delta = g.normal(0.0, 2.0, size=(n_channels, n_trials))
    rates_stim = rates_base + delta
    for j in real_idx:
        rates_stim[j] += effect
    pvals = np.array([stats.ttest_rel(rates_stim[j], rates_base[j]).pvalue
                      for j in range(n_channels)])
    return rates_base, rates_stim, pvals


def cr_generate(seed: int, variant: str) -> TaskInstance:
    g = _rng(f"{CR}:{seed}")
    if variant == "trapped":
        # Many channels, NO true effect; ensure >=2 raw hits but none survive BH.
        m, real, effect = 40, [], 0.0
        for _ in range(600):
            base, stim, p = _cr_pvals(g, m, CR_N_TRIALS, real, effect)
            if int((p < 0.05).sum()) >= 2 and not _bh(p, 0.05).any():
                break
        trap_type = TrapType.MULTIPLE_COMPARISONS
        trap_note = ("none of the channels are truly responsive, but several cross raw "
                     "p<0.05 by chance; none survive Benjamini-Hochberg FDR control")
    else:
        # A few channels with one clearly, genuinely responsive channel and no
        # chance hits among the rest.
        m, real, effect = 6, [0], 3.0
        for _ in range(600):
            base, stim, p = _cr_pvals(g, m, CR_N_TRIALS, real, effect)
            raw = set(np.where(p < 0.05)[0])
            if raw == {0} and _bh(p, 0.05)[0]:
                break
        trap_type = None
        trap_note = ""

    sig = _bh(p, 0.05)
    true_responsive = [f"ch{j:02d}" for j in np.where(sig)[0]]

    # rates.csv: one row per (channel, trial) with baseline and stimulus rates.
    lines = ["channel,trial,baseline_rate_hz,stimulus_rate_hz"]
    for j in range(m):
        for t in range(CR_N_TRIALS):
            lines.append(f"ch{j:02d},t{t:02d},{base[j, t]:.4f},{stim[j, t]:.4f}")
    rates_csv = "\n".join(lines) + "\n"

    prompt = (
        f"rates.csv reports per-trial baseline and stimulus firing rates (Hz) for {m} "
        f"recording channels ({CR_N_TRIALS} trials each). Which channels are genuinely "
        f"stimulus-responsive (a real difference between stimulus and baseline)? Test "
        f"each channel and report only the channels that remain significant after "
        f"appropriate control for testing many channels, at level 0.05."
    )
    return TaskInstance(
        task_id=_tid(CR, seed, variant), family=CR, domain="neuroscience",
        variant=variant, seed=seed, prompt=prompt,
        assets={"rates.csv": rates_csv},
        answer_fields={
            "responsive_channels": "list of genuinely responsive channel ids (empty if none)",
        },
        trap_type=trap_type, trap_note=trap_note,
        answer={"responsive_channels": true_responsive},
        grading={"trap_keywords": ["correct", "bonferroni", "fdr", "benjamini", "multiple compar",
                                   "false discovery", "adjust", "by chance", "family-wise",
                                   "familywise", "no channel", "none survive", "expected by chance"]},
    )


def cr_grade(inst, sub):
    got = sub.answers.get("responsive_channels") or []
    got = set(str(x).strip() for x in got) if isinstance(got, list) else set()
    correct = got == set(inst.answer["responsive_channels"])
    return standard_grade(inst, sub, correct_now=correct,
                          trap_keywords=inst.grading["trap_keywords"])


def _cr_read(view: AgentView):
    rows = [ln for ln in view.assets["rates.csv"].splitlines() if ln.strip()][1:]
    base: dict[str, list[float]] = {}
    stim: dict[str, list[float]] = {}
    for ln in rows:
        ch, _t, b, s = ln.split(",")
        base.setdefault(ch, []).append(float(b))
        stim.setdefault(ch, []).append(float(s))
    return base, stim


def _cr_solve(view: AgentView, careful: bool) -> Submission:
    base, stim = _cr_read(view)
    channels = sorted(base)
    p = np.array([stats.ttest_rel(stim[c], base[c]).pvalue for c in channels])
    if careful:
        sig = _bh(p, 0.05)
        n_raw = int((p < 0.05).sum())
        issues = []
        if n_raw > int(sig.sum()):
            issues.append(
                f"{n_raw} of {len(channels)} channels cross raw p<0.05, expected by chance "
                f"when testing many channels; applied Benjamini-Hochberg FDR control at 0.05")
        names = [channels[i] for i in np.where(sig)[0]]
    else:
        # Naive: report every channel below raw p<0.05, no correction.
        names = [channels[i] for i in np.where(p < 0.05)[0]]
    return Submission(answers={"responsive_channels": names},
                      issues_detected=issues if careful else [])


# --------------------------------------------------------------------------- #
# Family 3 — is a firing-rate difference real? (underpowered sample).
# --------------------------------------------------------------------------- #
# Two conditions, single neuron's firing rate across trials. The trapped twin
# has a tiny number of trials with no true effect but a chance p<0.05.
UP = "neuro.tuning_difference"
UP_N_MIN = 8                 # below this, a single significance test is untrustworthy


def up_generate(seed: int, variant: str) -> TaskInstance:
    g = _rng(f"{UP}:{seed}")
    if variant == "trapped":
        # Tiny sample, NO true effect, but a chance p<0.05.
        nper = 3
        for _ in range(4000):
            a = g.normal(12.0, 3.0, size=nper)
            b = g.normal(12.0, 3.0, size=nper)
            if stats.ttest_ind(a, b).pvalue < 0.05:
                break
        supports = False     # null effect + far too few trials: must not claim a difference
        trap_type = TrapType.UNDERPOWERED
        trap_note = (f"only n={nper} trials/condition with no true effect; a chance p<0.05 "
                     f"from this tiny sample must not be reported as a real difference")
    else:
        # Adequate sample with a genuine firing-rate difference.
        nper = 30
        a = g.normal(12.0, 3.0, size=nper)
        b = g.normal(15.5, 3.0, size=nper)
        supports = True
        trap_type = None
        trap_note = ""

    lines = ["condition,trial,firing_rate_hz"]
    for cond, arr in (("A", a), ("B", b)):
        for t, v in enumerate(arr):
            lines.append(f"{cond},t{t:02d},{v:.4f}")
    rates_csv = "\n".join(lines) + "\n"

    prompt = (
        "firing_rates.csv gives a single neuron's firing rate (Hz) on each trial under two "
        "conditions A and B. Is there real evidence that the neuron's firing rate differs "
        "between conditions (alpha=0.05)? Report whether the evidence supports a genuine "
        "difference, and judge whether the number of trials is adequate to draw that conclusion."
    )
    return TaskInstance(
        task_id=_tid(UP, seed, variant), family=UP, domain="neuroscience",
        variant=variant, seed=seed, prompt=prompt,
        assets={"firing_rates.csv": rates_csv},
        answer_fields={
            "supports_difference": "true/false: real evidence of a firing-rate difference",
            "n_trials_per_condition": "number of trials per condition",
        },
        trap_type=trap_type, trap_note=trap_note,
        answer={"supports_difference": supports},
        grading={"trap_keywords": ["underpower", "sample size", "too few", "too small",
                                   "power", "insufficient", "not enough", "n=3", "only 3",
                                   "three trials", "unreliable", "inadequate", "wide"]},
    )


def up_grade(inst, sub):
    correct = parse_bool(sub.answers.get("supports_difference")) == inst.answer["supports_difference"]
    return standard_grade(inst, sub, correct_now=correct,
                          trap_keywords=inst.grading["trap_keywords"])


def _up_solve(view: AgentView, careful: bool) -> Submission:
    rows = [ln for ln in view.assets["firing_rates.csv"].splitlines() if ln.strip()][1:]
    a, b = [], []
    for ln in rows:
        cond, _t, v = ln.split(",")
        (a if cond == "A" else b).append(float(v))
    a, b = np.array(a), np.array(b)
    nper = min(len(a), len(b))
    p = stats.ttest_ind(a, b).pvalue
    if careful and nper < UP_N_MIN:
        return Submission(
            answers={"supports_difference": False, "n_trials_per_condition": nper},
            issues_detected=[f"underpowered: only n={nper} trials/condition is too few to "
                             f"trust a single p-value; withholding the claim"])
    return Submission(answers={"supports_difference": bool(p < 0.05),
                               "n_trials_per_condition": nper})


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
for fam in [
    Family(DD, "neuroscience", "Neural decoding without double dipping",
           "Estimate decoding accuracy for the most discriminative neurons; the trapped "
           "twin has no true signal, so selecting and evaluating on the same trials inflates "
           "the accuracy far above the honest held-out estimate.",
           [TrapType.CIRCULAR_ANALYSIS], dd_generate, dd_grade,
           lambda v: _dd_solve(v, False), lambda v: _dd_solve(v, True)),
    Family(CR, "neuroscience", "Channel responsiveness with multiple comparisons",
           "Find genuinely stimulus-responsive channels among many; in the trapped twin "
           "none are responsive but several cross raw p<0.05 by chance and none survive "
           "FDR control.",
           [TrapType.MULTIPLE_COMPARISONS], cr_generate, cr_grade,
           lambda v: _cr_solve(v, False), lambda v: _cr_solve(v, True)),
    Family(UP, "neuroscience", "Firing-rate difference from an adequate sample",
           "Decide whether a neuron's firing rate differs between two conditions; the "
           "trapped twin is far too few trials with no true effect but a chance p<0.05.",
           [TrapType.UNDERPOWERED], up_generate, up_grade,
           lambda v: _up_solve(v, False), lambda v: _up_solve(v, True)),
]:
    registry.register(fam)
