"""Neuroscience domain — systems/electrophysiology methodology traps.

Migrated to the v2 contract; mirrors :mod:`shb.domains.chemistry` and
:mod:`shb.domains.statistics`. The three families are statistical analysis
pitfalls a competent electrophysiologist would catch. Per family:

* ``generate(seed, variant)`` is deterministic and **salted** (``np_seed`` /
  ``family_rng``). For the paired family (underpowered) the clean and trapped
  instances for a seed are genuine counterfactual twins — identical except the
  injected flaw — and store a shared ``base_signature`` so a test can verify it.
  The circular-analysis and multiple-comparisons families differ in the
  underlying signal/dimensionality between variants, so they are *not* pure
  twins and set ``paired=False`` (reported as robustness scenarios).
* ground truth is computed with the same numpy/scipy pipelines the reference
  solvers use, and stored hidden in ``answer``.
* ``grade`` builds an ``evidence_ok`` check from the hidden ``grading`` payload —
  one that verifies the agent found the *specific* planted flaw against ground
  truth, not satisfiable by guessing — and calls :func:`shb.utils.standard_grade`.
* ``ref_naive`` trusts the obvious (flawed) analysis and commits the fake
  science; ``ref_careful`` runs the correct analysis and emits a **structured,
  evidence-bearing** issue. Both read only the public :class:`AgentView`.
* each family supplies a neutral default (cued) prompt and an ``prompt_uncued``
  variant with no trap vocabulary, for the uncued track.

The traps, one distinct :class:`TrapType` per family:

* CIRCULAR_ANALYSIS — neural decoding by "double dipping". Single-unit responses
  (neurons x trials) under two conditions. The trapped twin has *no* true
  condition signal, so selecting the most discriminative neurons and scoring
  them on the *same* trials yields a spuriously high accuracy while an honest
  cross-validated estimate is at chance. Held-out vs in-sample is the tell.
* MULTIPLE_COMPARISONS — many channels are tested for a stimulus effect. The
  trapped twin has many channels, none responsive, but several cross raw p<0.05
  by chance and none survive Benjamini-Hochberg control. Differs in
  dimensionality, so ``paired=False``.
* UNDERPOWERED — decide whether a firing-rate difference is real. The trapped
  twin is the FIRST few trials of the clean larger sample (a subset), so only
  the (inadequate) trial count differs — a true twin. A chance p<0.05 from the
  tiny sample must not be reported as a real effect.
"""
from __future__ import annotations

import io
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .. import registry
from ..taxonomy import TrapType
from ..types import AgentView, Family, Submission, TaskInstance
from ..utils import (approx, base_signature, ev_near, ev_text, family_rng,
                     issue, np_seed, parse_bool, standard_grade)

# Tiny subsets and near-chance decoders emit numerical noise, not errors.
warnings.simplefilter("ignore")
np.seterr(all="ignore")


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


def _csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


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
#   Clean has a real condition signal; trapped has none. Because the underlying
#   signal differs between variants this is NOT a pure twin -> paired=False.
# --------------------------------------------------------------------------- #
DD = "neuro.decoding_circular"
N_UNITS = 200                # candidate neurons (a large pool to select from)
N_TRIALS_PER = 40            # trials per condition
DD_TOPK = 5                  # neurons the decoder is built from
DD_KFOLD = 5                 # cross-validation folds
DD_REPEATS = 20              # repeated CV; averages out fold noise near chance
DD_CV_KEY = f"{DD}:cv"       # fixed CV shuffle key; identical on generate/solve sides
DD_INFLATION = 0.10          # in-sample minus held-out gap that flags circularity


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


def _dd_cv_accuracy(resp, lab, topk, kfold, repeats) -> float:
    """Honest accuracy: repeated stratified k-fold CV with neuron selection
    *inside* each fold's training data. Averaging over repeats concentrates the
    estimate at chance when there is no signal, so the ground truth is stable.
    A fixed shuffle key makes the value identical on the generate and solve sides.
    """
    g = np.random.default_rng(np_seed(DD_CV_KEY, 0))
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


def dd_generate(seed: int, variant: str) -> TaskInstance:
    g = np.random.default_rng(np_seed(DD, seed))
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
    resp = np.round(np.concatenate([a, b], axis=1), 4)   # rounded == bytes the solver reads
    lab = np.array([0] * N_TRIALS_PER + [1] * N_TRIALS_PER)

    # Ground truth on the exact (rounded) data: the honest cross-validated
    # accuracy, plus the inflated in-sample value the double dip produces.
    honest_acc = _dd_cv_accuracy(resp, lab, DD_TOPK, DD_KFOLD, DD_REPEATS)
    insample_acc = _dd_insample_accuracy(resp, lab, DD_TOPK)

    if variant == "trapped":
        trap_type = TrapType.CIRCULAR_ANALYSIS
        trap_note = ("no true condition signal; selecting the most discriminative neurons "
                     "and scoring them on the SAME trials (double dipping) inflates the "
                     "apparent decoding accuracy far above the honest cross-validated value")
    else:
        trap_type, trap_note = None, ""

    # responses.csv: neurons in rows, single-trial responses in columns; each
    # column header carries the trial's condition (A or B).
    cols = [f"{'A' if l == 0 else 'B'}_t{j:02d}" for j, l in enumerate(lab)]
    lines = ["neuron," + ",".join(cols)]
    for i in range(resp.shape[0]):
        lines.append(f"n{i:03d}," + ",".join(f"{resp[i, j]:.4f}" for j in range(resp.shape[1])))
    responses_csv = "\n".join(lines) + "\n"

    cued = (
        f"responses.csv gives single-trial responses of {N_UNITS} neurons (neurons in rows, "
        f"trials in columns) under two conditions; each column header is the trial's condition "
        f"(A or B). Build a decoder from the {DD_TOPK} most condition-discriminative neurons and "
        f"report its classification accuracy (fraction of trials correctly labelled A vs B). "
        f"Report the accuracy that reflects how well the decoder generalizes to new trials, not "
        f"its fit on the same trials used to select the neurons.")
    uncued = (
        f"responses.csv gives single-trial responses of {N_UNITS} neurons (neurons in rows, "
        f"trials in columns) under two conditions; each column header is the trial's condition "
        f"(A or B). Build a decoder from the {DD_TOPK} most condition-discriminative neurons and "
        f"report its classification accuracy (fraction of trials correctly labelled A vs B).")
    return TaskInstance(
        task_id=_tid(DD, seed, variant), family=DD, domain="neuroscience",
        variant=variant, seed=seed, prompt=cued, assets={"responses.csv": responses_csv},
        answer_fields={"decoding_accuracy": "cross-validated decoding accuracy (fraction in [0,1])"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"decoding_accuracy": honest_acc},
        grading={"atol": 0.10, "honest_acc": honest_acc, "insample_acc": insample_acc,
                 "prompt_uncued": uncued})


def dd_grade(inst, sub):
    a = inst.answer
    answer_correct = approx(sub.answers.get("decoding_accuracy"), a["decoding_accuracy"],
                            rtol=0.0, atol=inst.grading["atol"])
    honest_acc = inst.grading["honest_acc"]
    insample_acc = inst.grading["insample_acc"]
    inflated = (insample_acc - honest_acc) > DD_INFLATION   # true only in the trapped twin

    def ev_ok(ev):
        # Detection means: report the held-out estimate (near the honest chance
        # value) AND the inflated in-sample value, demonstrating the double dip.
        # Both are specific per-instance numbers, not guessable; only valid when
        # the ground-truth gap is real (the trapped twin).
        if not inflated or not isinstance(ev, dict):
            return False
        held = ev.get("held_out_accuracy")
        ins = ev.get("in_sample_accuracy")
        if held is None or ins is None:
            return False
        return (approx(held, honest_acc, rtol=0.0, atol=0.08)
                and approx(ins, insample_acc, rtol=0.0, atol=0.08)
                and (float(ins) - float(held)) > DD_INFLATION)

    return standard_grade(inst, sub, answer_correct=answer_correct,
                          flaw_kind="circular_analysis", evidence_ok=ev_ok)


def _dd_solve(view: AgentView, careful: bool) -> Submission:
    _, hdr, resp = _read_matrix_csv(view.assets["responses.csv"])
    lab = np.array([0 if h.startswith("A") else 1 for h in hdr])
    insample = _dd_insample_accuracy(resp, lab, DD_TOPK)
    if careful:
        # Honest: cross-validate with neuron selection inside each fold, then
        # compare against the in-sample estimate to diagnose circularity.
        honest = _dd_cv_accuracy(resp, lab, DD_TOPK, DD_KFOLD, DD_REPEATS)
        issues = []
        if insample - honest > DD_INFLATION:
            issues.append(issue("circular_analysis",
                                held_out_accuracy=round(honest, 4),
                                in_sample_accuracy=round(insample, 4),
                                note="selecting the most discriminative neurons and scoring them "
                                     "on the same trials is circular (double dipping); held-out "
                                     "accuracy is near chance while the in-sample estimate is inflated"))
        return Submission(answers={"decoding_accuracy": round(honest, 4)},
                          issues=issues, confidence=0.9)
    # Naive: select the best neurons on all the trials and report their
    # separation on those very same trials -> double dipping.
    return Submission(answers={"decoding_accuracy": round(insample, 4)}, confidence=1.0)


# --------------------------------------------------------------------------- #
# Family 2 — channel responsiveness with uncorrected multiple comparisons.
#   The variants differ in the NUMBER of channels (dimensionality), so this is
#   NOT a pure twin: paired=False (like stats.multiple_comparisons).
# --------------------------------------------------------------------------- #
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
    g = np.random.default_rng(np_seed(CR, seed))
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
        trap_type, trap_note = None, ""

    sig = _bh(p, 0.05)
    n_raw = int((p < 0.05).sum())
    true_responsive = [f"ch{j:02d}" for j in np.where(sig)[0]]

    # rates.csv: one row per (channel, trial) with baseline and stimulus rates.
    lines = ["channel,trial,baseline_rate_hz,stimulus_rate_hz"]
    for j in range(m):
        for t in range(CR_N_TRIALS):
            lines.append(f"ch{j:02d},t{t:02d},{base[j, t]:.4f},{stim[j, t]:.4f}")
    rates_csv = "\n".join(lines) + "\n"

    cued = (
        f"rates.csv reports per-trial baseline and stimulus firing rates (Hz) for {m} recording "
        f"channels ({CR_N_TRIALS} trials each). Which channels are genuinely stimulus-responsive "
        f"(a real difference between stimulus and baseline)? Test each channel and report only the "
        f"channels that remain significant after appropriate control for testing many channels, "
        f"at level 0.05.")
    uncued = (
        f"rates.csv reports per-trial baseline and stimulus firing rates (Hz) for {m} recording "
        f"channels ({CR_N_TRIALS} trials each). Which channels are genuinely stimulus-responsive, "
        f"at level 0.05?")
    return TaskInstance(
        task_id=_tid(CR, seed, variant), family=CR, domain="neuroscience",
        variant=variant, seed=seed, prompt=cued, assets={"rates.csv": rates_csv},
        answer_fields={"responsive_channels": "list of genuinely responsive channel ids (empty if none)"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"responsive_channels": true_responsive},
        grading={"n_corrected_truth": int(sig.sum()), "n_raw_truth": n_raw,
                 "prompt_uncued": uncued})


def cr_grade(inst, sub):
    got = sub.answers.get("responsive_channels") or []
    got = set(str(x).strip() for x in got) if isinstance(got, list) else set()
    answer_correct = got == set(inst.answer["responsive_channels"])
    n_corrected_truth = inst.grading["n_corrected_truth"]
    n_raw_truth = inst.grading["n_raw_truth"]

    def ev_ok(ev):
        # Detection on the trap means: raw-significant channels do NOT survive
        # correction. Only valid when the ground truth is exactly that (no
        # channel survives, >=2 raw hits).
        if not (n_corrected_truth == 0 and n_raw_truth >= 2) or not isinstance(ev, dict):
            return False
        n_corr = ev.get("n_corrected_significant")
        n_raw = ev.get("n_raw_significant")
        return (n_corr is not None and int(n_corr) == 0
                and n_raw is not None and int(n_raw) == n_raw_truth)

    return standard_grade(inst, sub, answer_correct=answer_correct,
                          flaw_kind="multiple_comparisons", evidence_ok=ev_ok)


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
    n_raw = int((p < 0.05).sum())
    if careful:
        sig = _bh(p, 0.05)
        names = [channels[i] for i in np.where(sig)[0]]
        issues = ([issue("multiple_comparisons", n_raw_significant=n_raw,
                         n_corrected_significant=int(sig.sum()),
                         note=f"{n_raw} of {len(channels)} channels cross raw p<0.05, expected by "
                              f"chance when testing many channels; applied Benjamini-Hochberg FDR control")]
                  if n_raw > int(sig.sum()) else [])
        return Submission(answers={"responsive_channels": names}, issues=issues, confidence=0.9)
    names = [channels[i] for i in np.where(p < 0.05)[0]]   # naive: uncorrected raw hits
    return Submission(answers={"responsive_channels": names}, confidence=1.0)


# --------------------------------------------------------------------------- #
# Family 3 — is a firing-rate difference real? (underpowered sample).
#   Trapped is the FIRST N_SMALL trials of the clean N_FULL sample (a subset),
#   so ONLY the trial count differs (a true twin): paired=True.
# --------------------------------------------------------------------------- #
UP = "neuro.tuning_difference"
UP_N_FULL = 30
UP_N_SMALL = 4
UP_N_MIN = 10                # below this many trials/condition, a p-value is not trustworthy


def up_generate(seed: int, variant: str) -> TaskInstance:
    g = np.random.default_rng(np_seed(UP, seed))
    # One underlying real firing-rate difference and one pair of full samples,
    # shared by both twins. Accept the draw only when the FULL sample is clearly
    # significant AND its first-N_SMALL subset is *also* significant — so the
    # trapped subset traps a naive tester (it sees p<0.05) even though the trial
    # count is far too small to support the claim.
    for _ in range(4000):
        a = g.normal(12.0, 3.0, size=UP_N_FULL)
        b = g.normal(15.5, 3.0, size=UP_N_FULL)
        p_full = stats.ttest_ind(a, b).pvalue
        p_small = stats.ttest_ind(a[:UP_N_SMALL], b[:UP_N_SMALL]).pvalue
        if p_full < 0.01 and p_small < 0.05:
            break

    if variant == "trapped":
        nper, supports = UP_N_SMALL, False   # too few trials: must NOT claim a difference
        trap_type = TrapType.UNDERPOWERED
        trap_note = (f"only n={UP_N_SMALL} trials/condition; a chance p<0.05 from a sample this "
                     f"small must not be reported as a real firing-rate difference")
    else:
        nper, supports = UP_N_FULL, True     # adequate sample; the effect is real and supported
        trap_type, trap_note = None, ""
    a_v, b_v = a[:nper], b[:nper]

    df = pd.DataFrame({"condition": ["A"] * nper + ["B"] * nper,
                       "trial": [f"t{t:02d}" for t in range(nper)] * 2,
                       "firing_rate_hz": np.round(np.concatenate([a_v, b_v]), 4)})
    cued = (
        "firing_rates.csv gives a single neuron's firing rate (Hz) on each trial under two "
        "conditions A and B. Is there real evidence that the neuron's firing rate differs between "
        "conditions (alpha=0.05)? Report whether the evidence supports a genuine difference, and "
        "judge whether the number of trials is adequate to draw that conclusion.")
    uncued = (
        "firing_rates.csv gives a single neuron's firing rate (Hz) on each trial under two "
        "conditions A and B. Is there real evidence that the neuron's firing rate differs between "
        "conditions (alpha=0.05)?")
    return TaskInstance(
        task_id=_tid(UP, seed, variant), family=UP, domain="neuroscience",
        variant=variant, seed=seed, prompt=cued, assets={"firing_rates.csv": _csv(df)},
        answer_fields={"supports_difference": "true/false: real evidence of a firing-rate difference",
                       "n_trials_per_condition": "number of trials per condition"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"supports_difference": supports, "n_trials_per_condition": nper},
        grading={"n_bad": UP_N_SMALL, "n_min": UP_N_MIN,
                 "base_signature": base_signature(np.round(a, 6).tobytes(), np.round(b, 6).tobytes()),
                 "prompt_uncued": uncued})


def up_grade(inst, sub):
    answer_correct = parse_bool(sub.answers.get("supports_difference")) == inst.answer["supports_difference"]
    n_bad, n_min = inst.grading["n_bad"], inst.grading["n_min"]
    is_trapped = inst.is_trapped

    def ev_ok(ev):
        if not is_trapped:
            return False
        n_rep = ev.get("n_trials_per_condition") if isinstance(ev, dict) else None
        # Must identify the specific inadequate trial count (n=4 < the adequacy floor).
        named = (n_rep is not None and int(n_rep) == n_bad) or ev_near(ev, n_bad, rtol=0.0)
        return bool(named) and n_bad < n_min

    return standard_grade(inst, sub, answer_correct=answer_correct,
                          flaw_kind="underpowered", evidence_ok=ev_ok)


def _up_solve(view: AgentView, careful: bool) -> Submission:
    df = pd.read_csv(io.StringIO(view.assets["firing_rates.csv"]))
    a = df[df.condition == "A"].firing_rate_hz.to_numpy()
    b = df[df.condition == "B"].firing_rate_hz.to_numpy()
    nper = min(len(a), len(b))
    p = stats.ttest_ind(a, b).pvalue
    if careful and nper < UP_N_MIN:
        # A power/precision argument: the per-condition CI half-width on the mean
        # is huge relative to the effect, so a p<0.05 here cannot support a real
        # difference.
        ci_halfwidth = float(stats.t.ppf(0.975, nper - 1) * np.std(a, ddof=1) / np.sqrt(nper))
        return Submission(answers={"supports_difference": False, "n_trials_per_condition": nper},
                          issues=[issue("underpowered", n_trials_per_condition=nper,
                                        ci_halfwidth_condA=round(ci_halfwidth, 3),
                                        note=f"n={nper} trials/condition is below the adequacy floor; "
                                             f"a single p-value from this tiny sample is not trustworthy")],
                          confidence=0.9)
    return Submission(answers={"supports_difference": bool(p < 0.05), "n_trials_per_condition": nper},
                      confidence=0.9 if careful else 1.0)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
for fam in [
    Family(DD, "neuroscience", "Neural decoding without double dipping",
           "Estimate decoding accuracy for the most discriminative neurons; the trapped twin has "
           "no true signal, so selecting and evaluating on the same trials inflates the accuracy "
           "far above the honest held-out estimate. Variants differ in the underlying signal, so "
           "this is not a pure twin.",
           [TrapType.CIRCULAR_ANALYSIS], "circular_analysis", dd_generate, dd_grade,
           lambda v: _dd_solve(v, False), lambda v: _dd_solve(v, True), paired=False),
    Family(CR, "neuroscience", "Channel responsiveness with multiple comparisons",
           "Find genuinely stimulus-responsive channels among many; in the trapped twin none are "
           "responsive but several cross raw p<0.05 by chance and none survive FDR control. "
           "Variants differ in dimensionality, so this is not a pure twin.",
           [TrapType.MULTIPLE_COMPARISONS], "multiple_comparisons", cr_generate, cr_grade,
           lambda v: _cr_solve(v, False), lambda v: _cr_solve(v, True), paired=False),
    Family(UP, "neuroscience", "Firing-rate difference from an adequate sample",
           "Decide whether a neuron's firing rate differs between two conditions; the trapped "
           "sample is the first n=4 trials of the clean n=30 sample, so only the (inadequate) "
           "trial count differs.",
           [TrapType.UNDERPOWERED], "underpowered", up_generate, up_grade,
           lambda v: _up_solve(v, False), lambda v: _up_solve(v, True), paired=True),
]:
    registry.register(fam)
