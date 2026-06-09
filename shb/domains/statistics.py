"""Statistics / data-analysis domain — the universal fake-science substrate.

Migrated to the v2 contract; mirrors :mod:`shb.domains.chemistry`. These traps
are pure methodology errors that apply to every empirical science: target
leakage, Simpson's paradox / confounding, uncorrected multiple comparisons, and
conclusions from underpowered samples. Per family:

* ``generate(seed, variant)`` is deterministic and **salted** (``family_rng`` /
  ``np_seed``). For the paired families the clean and trapped instances for a
  seed are genuine counterfactual twins — identical except the injected flaw —
  and store a shared ``base_signature`` so a test can verify it. Multiple
  comparisons differs in dimensionality between variants, so it is *not* a true
  twin and sets ``paired=False`` (reported as a robustness scenario).
* ground truth is computed with the same numpy/scipy/sklearn pipelines the
  reference solvers use, and stored hidden in ``answer``.
* ``grade`` builds an ``evidence_ok`` check from the hidden ``grading`` payload —
  one that verifies the agent found the *specific* planted flaw against ground
  truth, not satisfiable by guessing — and calls :func:`shb.utils.standard_grade`.
* ``ref_naive`` trusts inputs and answers confidently (commits fake science);
  ``ref_careful`` validates, corrects, and emits a **structured, evidence-bearing**
  issue. Both read only the public :class:`AgentView`.
* each family supplies a neutral default (cued) prompt and an ``prompt_uncued``
  variant with no trap vocabulary, for the uncued track.
"""
from __future__ import annotations

import io
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import registry
from ..taxonomy import TrapType
from ..types import AgentView, Family, Submission, TaskInstance
from ..utils import (approx, base_signature, ev_near, ev_text, family_rng,
                     issue, np_seed, parse_bool, standard_grade)

# Underpowered subsets and tiny logistic fits emit numerical noise, not errors.
warnings.simplefilter("ignore", category=ConvergenceWarning)
np.seterr(all="ignore")


def _cv_auc(X: np.ndarray, y: np.ndarray) -> float:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean())


def _single_auc(col: np.ndarray, y: np.ndarray) -> float:
    a = roc_auc_score(y, col)
    return max(a, 1 - a)


def _bh(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    sig = np.zeros(m, bool)
    if passed.any():
        k = np.max(np.where(passed)[0]) + 1
        sig[order[:k]] = True
    return sig


def _tid(fam: str, seed: int, variant: str) -> str:
    return f"{fam}/seed={seed}/{variant}"


def _csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


# --------------------------------------------------------------------------- #
# Family 1 — target leakage; trapped adds ONE leaky column to the same data.
# --------------------------------------------------------------------------- #
LK = "stats.data_leakage"


def lk_generate(seed: int, variant: str) -> TaskInstance:
    g = np.random.default_rng(np_seed(LK, seed))
    n, k = 320, 4
    X = g.normal(size=(n, k))                       # the legitimate features (shared by both twins)
    w = g.normal(size=k)
    logits = X @ w + g.normal(scale=0.5, size=n)
    y = (logits > np.median(logits)).astype(int)
    cols = {f"f{i}": X[:, i] for i in range(k)}

    leaky: list[str] = []
    if variant == "trapped":
        t = np.random.default_rng(np_seed(LK, seed, trap=True))
        # The ONLY difference from the clean twin: one extra column that is the
        # label plus a hair of noise — near-perfect single-feature separation.
        cols[f"f{k}"] = y + t.normal(scale=0.01, size=n)
        leaky = [f"f{k}"]

    df = pd.DataFrame({**cols, "y": y})
    legit = [c for c in df.columns if c not in ("y", *leaky)]
    honest_auc = _cv_auc(df[legit].to_numpy(), y)   # identical for both twins (same legit features)

    cued = ("data.csv has feature columns f* and a binary label y. Report the 5-fold "
            "cross-validated ROC-AUC achievable with the legitimate features, and list any "
            "feature that leaks the label (a feature you must exclude to get an honest estimate). "
            "Check whether any single feature separates the label implausibly well before using it.")
    uncued = ("data.csv has feature columns f* and a binary label y. Report the 5-fold "
              "cross-validated ROC-AUC achievable with the legitimate features, and list any "
              "feature that leaks the label.")
    return TaskInstance(
        task_id=_tid(LK, seed, variant), family=LK, domain="statistics",
        variant=variant, seed=seed, prompt=cued, assets={"data.csv": _csv(df)},
        answer_fields={"leaky_features": "list of leaking column names (empty list if none)",
                       "honest_auc": "5-fold CV ROC-AUC using only legitimate features"},
        trap_type=TrapType.DATA_LEAKAGE if leaky else None,
        trap_note=(f"{leaky[0]} is the label plus tiny noise (target leakage)" if leaky else ""),
        answer={"leaky_features": leaky, "honest_auc": honest_auc},
        grading={"rtol": 0.12, "leaky_features": leaky,
                 "base_signature": base_signature(np.round(X, 6).tobytes(), y.tobytes()),
                 "prompt_uncued": uncued})


def lk_grade(inst, sub):
    got = sub.answers.get("leaky_features") or []
    got = [str(x).strip() for x in got] if isinstance(got, list) else []
    set_ok = set(got) == set(inst.answer["leaky_features"])
    auc_ok = approx(sub.answers.get("honest_auc"), inst.answer["honest_auc"], rtol=inst.grading["rtol"])
    answer_correct = set_ok and auc_ok
    truth = set(inst.grading.get("leaky_features", []))

    def ev_ok(ev):
        if isinstance(ev, dict) and "leaky_features" in ev:
            rep = set(str(x).strip() for x in ev["leaky_features"])
        else:
            rep = set(w for w in ev_text(ev).replace(",", " ").split() if w.startswith("f") and w[1:].isdigit())
        return rep == truth and len(truth) > 0

    return standard_grade(inst, sub, answer_correct=answer_correct,
                          flaw_kind="data_leakage", evidence_ok=ev_ok)


def _lk_solve(view: AgentView, careful: bool) -> Submission:
    df = pd.read_csv(io.StringIO(view.assets["data.csv"]))
    y = df["y"].to_numpy()
    feats = [c for c in df.columns if c != "y"]
    leaky, issues = [], []
    if careful:
        for c in feats:
            if _single_auc(df[c].to_numpy(), y) > 0.985:    # one feature alone all-but-perfect: leak
                leaky.append(c)
                issues.append(issue("data_leakage", leaky_features=[c],
                                    single_feature_auc=round(_single_auc(df[c].to_numpy(), y), 4)))
    use = [c for c in feats if c not in leaky]
    auc = _cv_auc(df[use].to_numpy(), y)
    return Submission(answers={"leaky_features": leaky, "honest_auc": round(auc, 4)},
                      issues=issues if careful else [], confidence=0.9 if careful else 1.0)


# --------------------------------------------------------------------------- #
# Family 2 — Simpson's paradox; twins share the per-stratum RATES and differ
#   only in the group-size allocation that does/doesn't induce the reversal.
# --------------------------------------------------------------------------- #
SP = "stats.simpson"


def sp_generate(seed: int, variant: str) -> TaskInstance:
    g = family_rng(SP, seed)
    # Per-stratum success RATES, shared by both twins. B is better than A in EACH
    # stratum (the genuine within-stratum truth), so the correct answer is the
    # same for both: A is NOT better. Only the allocation of group sizes differs.
    s0 = (round(0.80 + g.uniform(-0.02, 0.02), 3), round(0.90 + g.uniform(-0.02, 0.02), 3))  # (A,B) rates
    s1 = (round(0.30 + g.uniform(-0.02, 0.02), 3), round(0.40 + g.uniform(-0.02, 0.02), 3))
    rates = [s0, s1]

    if variant == "trapped":
        # Allocation engineered so the POOLED rate reverses to A>B (Simpson's paradox):
        # A's mass sits in the high-rate stratum, B's in the low-rate stratum.
        alloc = [(200, 50), (40, 180)]
    else:
        # Balanced allocation: the pooled rate keeps the within-stratum direction (B>A).
        alloc = [(120, 120), (110, 110)]

    rows = []
    for i, ((ar, br), (an, bn)) in enumerate(zip(rates, alloc)):
        rows.append((f"S{i}", "A", an, int(round(ar * an))))
        rows.append((f"S{i}", "B", bn, int(round(br * bn))))
    df = pd.DataFrame(rows, columns=["stratum", "treatment", "n", "successes"])

    _g = df.groupby("treatment")[["successes", "n"]].sum()
    pooled = _g["successes"] / _g["n"]
    pooled_A_better = bool(pooled["A"] > pooled["B"])
    strat_A_better = False                          # B is better in every stratum, by construction
    trap = pooled_A_better != strat_A_better        # paradox present only in the trapped allocation

    cued = ("results.csv reports successes out of n for treatments A and B across patient "
            "strata. Does treatment A have a higher success rate than treatment B? Account for "
            "the stratum structure rather than only the pooled totals.")
    uncued = ("results.csv reports successes out of n for treatments A and B across patient "
              "strata. Does treatment A have a higher success rate than treatment B?")
    # Per-stratum (A,B) rates for the grader to verify the reversal claim against.
    strat_rates = [(round(ar, 4), round(br, 4)) for (ar, br) in rates]
    return TaskInstance(
        task_id=_tid(SP, seed, variant), family=SP, domain="statistics",
        variant=variant, seed=seed, prompt=cued, assets={"results.csv": _csv(df)},
        answer_fields={"A_better": "true if treatment A is genuinely better, else false"},
        trap_type=TrapType.CONFOUNDING if trap else None,
        trap_note=("Simpson's paradox: pooled comparison reverses the within-stratum truth" if trap else ""),
        answer={"A_better": strat_A_better},
        grading={"strat_rates": strat_rates, "pooled_A_better": pooled_A_better,
                 "base_signature": base_signature(tuple(strat_rates)),
                 "prompt_uncued": uncued})


def sp_grade(inst, sub):
    answer_correct = parse_bool(sub.answers.get("A_better")) == inst.answer["A_better"]
    truth_rates = inst.grading["strat_rates"]
    pooled_A_better = inst.grading["pooled_A_better"]

    def ev_ok(ev):
        if not pooled_A_better:                     # only the trapped (reversed) twin is detectable
            return False
        if not isinstance(ev, dict):
            return False
        rep = ev.get("stratum_rates")
        # Must reproduce each stratum's (A,B) rates AND show B beats A in every one.
        if not isinstance(rep, (list, tuple)) or len(rep) != len(truth_rates):
            return False
        for (ra, rb), (tra, trb) in zip(rep, truth_rates):
            if not (approx(ra, tra, rtol=0.05) and approx(rb, trb, rtol=0.05)):
                return False
            if not (float(rb) > float(ra)):         # B better within the stratum
                return False
        return bool(ev.get("pooled_A_better")) is True

    return standard_grade(inst, sub, answer_correct=answer_correct,
                          flaw_kind="confounding", evidence_ok=ev_ok)


def _sp_rates(df: pd.DataFrame):
    rates = []
    for s in sorted(df["stratum"].unique()):
        d = df[df.stratum == s]
        ra = d[d.treatment == "A"].successes.sum() / d[d.treatment == "A"].n.sum()
        rb = d[d.treatment == "B"].successes.sum() / d[d.treatment == "B"].n.sum()
        rates.append((round(float(ra), 4), round(float(rb), 4)))
    return rates


def _sp_solve(view: AgentView, careful: bool) -> Submission:
    df = pd.read_csv(io.StringIO(view.assets["results.csv"]))
    _g = df.groupby("treatment")[["successes", "n"]].sum()
    pooled = _g["successes"] / _g["n"]
    pooled_A_better = bool(pooled["A"] > pooled["B"])
    if not careful:
        return Submission(answers={"A_better": pooled_A_better}, confidence=1.0)
    rates = _sp_rates(df)
    strat_A_better = (np.mean([ra - rb for ra, rb in rates]) > 0)
    issues = []
    if strat_A_better != pooled_A_better:
        issues.append(issue("confounding", stratum_rates=rates, pooled_A_better=pooled_A_better))
    return Submission(answers={"A_better": bool(strat_A_better)},
                      issues=issues, confidence=0.9)


# --------------------------------------------------------------------------- #
# Family 3 — uncorrected multiple comparisons. The variants differ in
#   dimensionality (#features), so this is NOT a pure twin: paired=False.
# --------------------------------------------------------------------------- #
MC = "stats.multiple_comparisons"


def _feature_pvals(g, m, n, real_idx):
    grp = np.array([0] * (n // 2) + [1] * (n - n // 2))
    cols, pvals = {}, []
    for j in range(m):
        x = g.normal(size=n)
        if j in real_idx:
            x[grp == 1] += 1.0
        cols[f"f{j}"] = x
        pvals.append(stats.ttest_ind(x[grp == 0], x[grp == 1]).pvalue)
    return grp, cols, np.array(pvals)


def mc_generate(seed: int, variant: str) -> TaskInstance:
    g = np.random.default_rng(np_seed(MC, seed))
    if variant == "trapped":
        m, n, real = 40, 60, []
        for _ in range(400):            # ensure >=2 raw hits but none survive BH correction
            grp, cols, p = _feature_pvals(g, m, n, real)
            if (p < 0.05).sum() >= 2 and not _bh(p).any():
                break
    else:
        m, n, real = 6, 60, [0]
        for _ in range(400):            # ensure the one real feature is the only raw hit and survives
            grp, cols, p = _feature_pvals(g, m, n, real)
            raw = set(np.where(p < 0.05)[0])
            if raw == {0} and _bh(p)[0]:
                break
    names = list(cols.keys())
    sig = _bh(p, 0.05)
    n_raw = int((p < 0.05).sum())
    df = pd.DataFrame({**cols, "group": grp})
    cued = (f"data.csv has {m} feature columns f* and a binary group label. Which features are "
            f"genuinely associated with the group? Test each feature and report only those that "
            f"remain significant after appropriate multiple-comparison control at level 0.05.")
    uncued = (f"data.csv has {m} feature columns f* and a binary group label. Which features are "
              f"genuinely associated with the group, at level 0.05?")
    return TaskInstance(
        task_id=_tid(MC, seed, variant), family=MC, domain="statistics",
        variant=variant, seed=seed, prompt=cued, assets={"data.csv": _csv(df)},
        answer_fields={"significant_features": "list of feature names truly associated (after correction)"},
        trap_type=TrapType.MULTIPLE_COMPARISONS if variant == "trapped" else None,
        trap_note=("several features cross raw p<0.05 by chance; none survive FDR control"
                   if variant == "trapped" else ""),
        answer={"significant_features": [names[i] for i in np.where(sig)[0]]},
        grading={"n_corrected_truth": int(sig.sum()), "n_raw_truth": n_raw,
                 "prompt_uncued": uncued})


def mc_grade(inst, sub):
    got = sub.answers.get("significant_features") or []
    got = set(str(x).strip() for x in got) if isinstance(got, list) else set()
    answer_correct = got == set(inst.answer["significant_features"])
    n_corrected_truth = inst.grading["n_corrected_truth"]
    n_raw_truth = inst.grading["n_raw_truth"]

    def ev_ok(ev):
        # Detection on the trap means: raw-significant hits do NOT survive correction.
        # Only valid when the ground truth is exactly that (no feature survives, >=2 raw).
        if not (n_corrected_truth == 0 and n_raw_truth >= 2) or not isinstance(ev, dict):
            return False
        n_corr = ev.get("n_corrected_significant")
        n_raw = ev.get("n_raw_significant")
        return (n_corr is not None and int(n_corr) == 0
                and n_raw is not None and int(n_raw) == n_raw_truth)

    return standard_grade(inst, sub, answer_correct=answer_correct,
                          flaw_kind="multiple_comparisons", evidence_ok=ev_ok)


def _mc_solve(view: AgentView, careful: bool) -> Submission:
    df = pd.read_csv(io.StringIO(view.assets["data.csv"]))
    grp = df["group"].to_numpy()
    feats = [c for c in df.columns if c != "group"]
    p = np.array([stats.ttest_ind(df[c][grp == 0], df[c][grp == 1]).pvalue for c in feats])
    n_raw = int((p < 0.05).sum())
    if careful:
        sig = _bh(p, 0.05)
        names = [feats[i] for i in np.where(sig)[0]]
        issues = ([issue("multiple_comparisons", n_raw_significant=n_raw,
                         n_corrected_significant=int(sig.sum()))]
                  if n_raw > int(sig.sum()) else [])
        return Submission(answers={"significant_features": names}, issues=issues, confidence=0.9)
    names = [feats[i] for i in np.where(p < 0.05)[0]]   # naive: uncorrected raw hits
    return Submission(answers={"significant_features": names}, confidence=1.0)


# --------------------------------------------------------------------------- #
# Family 4 — underpowered conclusion; trapped is the FIRST n=4 of the clean
#   n=30 sample (a subset), so ONLY the sample size differs (true twin).
# --------------------------------------------------------------------------- #
UP = "stats.underpowered"
N_FULL = 30
N_SMALL = 4
N_MIN = 10          # below this many per group, a p-value is not trustworthy


def up_generate(seed: int, variant: str) -> TaskInstance:
    g = np.random.default_rng(np_seed(UP, seed))
    # One underlying real effect and one pair of full samples, shared by both twins.
    # Accept the draw only when the FULL n=30 sample is clearly significant AND its
    # first-4 subset is *also* significant — so the trapped subset traps a naive
    # tester (it sees p<0.05) even though n=4 is far too small to support the claim.
    for _ in range(4000):
        a = g.normal(loc=0.0, size=N_FULL)
        b = g.normal(loc=0.9, size=N_FULL)
        p_full = stats.ttest_ind(a, b).pvalue
        p_small = stats.ttest_ind(a[:N_SMALL], b[:N_SMALL]).pvalue
        if p_full < 0.01 and p_small < 0.05:
            break

    if variant == "trapped":
        nper, supports = N_SMALL, False     # n=4: underpowered, must NOT claim a difference
    else:
        nper, supports = N_FULL, True       # n=30: adequate, the effect is real and supported
    a_v, b_v = a[:nper], b[:nper]

    df = pd.DataFrame({"group": ["A"] * nper + ["B"] * nper,
                       "value": np.concatenate([a_v, b_v])})
    cued = ("measurements.csv has values for groups A and B. Is there real evidence that the "
            "two groups differ (alpha=0.05)? Report whether the evidence supports a genuine "
            "difference, and judge whether the sample is adequate to draw that conclusion.")
    uncued = ("measurements.csv has values for groups A and B. Is there real evidence that the "
              "two groups differ (alpha=0.05)? Report whether the evidence supports a difference.")
    return TaskInstance(
        task_id=_tid(UP, seed, variant), family=UP, domain="statistics",
        variant=variant, seed=seed, prompt=cued, assets={"measurements.csv": _csv(df)},
        answer_fields={"supports_difference": "true/false",
                       "n_per_group": "sample size per group"},
        trap_type=TrapType.UNDERPOWERED if variant == "trapped" else None,
        trap_note=(f"n={N_SMALL}/group: a chance p<0.05 from a sample this small must not be trusted"
                   if variant == "trapped" else ""),
        answer={"supports_difference": supports, "n_per_group": nper},
        grading={"n_bad": N_SMALL, "n_min": N_MIN,
                 "base_signature": base_signature(np.round(a, 6).tobytes(), np.round(b, 6).tobytes()),
                 "prompt_uncued": uncued})


def up_grade(inst, sub):
    answer_correct = parse_bool(sub.answers.get("supports_difference")) == inst.answer["supports_difference"]
    n_bad, n_min = inst.grading["n_bad"], inst.grading["n_min"]
    is_trapped = inst.is_trapped

    def ev_ok(ev):
        if not is_trapped:
            return False
        n_rep = ev.get("n_per_group") if isinstance(ev, dict) else None
        # Must identify the specific inadequate sample size (n=4 < the adequacy floor).
        named = (n_rep is not None and int(n_rep) == n_bad) or ev_near(ev, n_bad, rtol=0.0)
        return bool(named) and n_bad < n_min

    return standard_grade(inst, sub, answer_correct=answer_correct,
                          flaw_kind="underpowered", evidence_ok=ev_ok)


def _up_solve(view: AgentView, careful: bool) -> Submission:
    df = pd.read_csv(io.StringIO(view.assets["measurements.csv"]))
    a = df[df.group == "A"].value.to_numpy()
    b = df[df.group == "B"].value.to_numpy()
    nper = min(len(a), len(b))
    p = stats.ttest_ind(a, b).pvalue
    if careful and nper < N_MIN:
        # A power/precision argument: the per-group CI half-width on the mean is huge
        # relative to the effect, so a p<0.05 here cannot support a real difference.
        ci_halfwidth = float(stats.t.ppf(0.975, nper - 1) * np.std(a, ddof=1) / np.sqrt(nper))
        return Submission(answers={"supports_difference": False, "n_per_group": nper},
                          issues=[issue("underpowered", n_per_group=nper,
                                        ci_halfwidth_groupA=round(ci_halfwidth, 3),
                                        note=f"n={nper}/group below adequacy floor; p-value not trustworthy")],
                          confidence=0.9)
    return Submission(answers={"supports_difference": bool(p < 0.05), "n_per_group": nper},
                      confidence=0.9 if careful else 1.0)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
for fam in [
    Family(LK, "statistics", "Honest AUC with target leakage",
           "Estimate classifier performance; the trapped twin adds one leaking feature "
           "(only that extra column differs) that must be found and excluded.",
           [TrapType.DATA_LEAKAGE], "data_leakage", lk_generate, lk_grade,
           lambda v: _lk_solve(v, False), lambda v: _lk_solve(v, True), paired=True),
    Family(SP, "statistics", "Simpson's paradox",
           "Compare two treatments where the trapped allocation makes pooling reverse the "
           "within-stratum truth; twins share the per-stratum rates.",
           [TrapType.CONFOUNDING], "confounding", sp_generate, sp_grade,
           lambda v: _sp_solve(v, False), lambda v: _sp_solve(v, True), paired=True),
    Family(MC, "statistics", "Multiple comparisons",
           "Find truly associated features among many; raw p<0.05 hits arise by chance. "
           "Variants differ in dimensionality, so this is not a pure twin.",
           [TrapType.MULTIPLE_COMPARISONS], "multiple_comparisons", mc_generate, mc_grade,
           lambda v: _mc_solve(v, False), lambda v: _mc_solve(v, True), paired=False),
    Family(UP, "statistics", "Underpowered conclusion",
           "Decide if two groups differ; the trapped sample is the first n=4 of the clean "
           "n=30 sample, so only the (inadequate) sample size differs.",
           [TrapType.UNDERPOWERED], "underpowered", up_generate, up_grade,
           lambda v: _up_solve(v, False), lambda v: _up_solve(v, True), paired=True),
]:
    registry.register(fam)
