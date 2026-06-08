"""Statistics / data-analysis domain — the universal fake-science substrate.

These traps are pure methodology errors and apply to every empirical science:
target leakage, Simpson's paradox / confounding, uncorrected multiple
comparisons, and conclusions from underpowered samples. Ground truth is computed
with the same numpy/scipy/sklearn pipelines the reference solvers use, so the
benchmark grades against an exact, deterministic reference.
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import registry
from ..taxonomy import TrapType
from ..types import AgentView, Family, Submission, TaskInstance
from ..utils import approx, parse_bool, standard_grade, to_float, jaccard


def _rng(seed_str: str) -> np.random.Generator:
    # Hash the full string: a prefix-sharing scheme (e.g. low bytes mod 2**32)
    # would collapse every seed of a family to one identical problem.
    digest = hashlib.sha256(seed_str.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


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
# Family 1 — target leakage in a classifier.
# --------------------------------------------------------------------------- #
LK = "stats.data_leakage"


def lk_generate(seed: int, variant: str) -> TaskInstance:
    g = _rng(f"{LK}:{seed}")
    n, k = 320, 4
    X = g.normal(size=(n, k))
    w = g.normal(size=k)
    logits = X @ w + g.normal(scale=0.5, size=n)
    y = (logits > np.median(logits)).astype(int)
    cols = {f"f{i}": X[:, i] for i in range(k)}

    leaky = []
    if variant == "trapped":
        # f4 is the label with a hair of noise: near-perfect single-feature separation.
        cols[f"f{k}"] = y + g.normal(scale=0.01, size=n)
        leaky = [f"f{k}"]

    df = pd.DataFrame({**cols, "y": y})
    legit = [c for c in df.columns if c not in ("y", *leaky)]
    honest_auc = _cv_auc(df[legit].to_numpy(), y)

    prompt = ("data.csv has feature columns f* and a binary label y. Report the 5-fold "
              "cross-validated ROC-AUC achievable with the legitimate features, and list any "
              "feature that leaks the label (a feature you must exclude to get an honest estimate).")
    return TaskInstance(
        task_id=_tid(LK, seed, variant), family=LK, domain="statistics",
        variant=variant, seed=seed, prompt=prompt, assets={"data.csv": _csv(df)},
        answer_fields={"leaky_features": "list of leaking column names (empty list if none)",
                       "honest_auc": "5-fold CV ROC-AUC using only legitimate features"},
        trap_type=TrapType.DATA_LEAKAGE if leaky else None,
        trap_note=(f"{leaky[0]} is the label plus tiny noise (target leakage)" if leaky else ""),
        answer={"leaky_features": leaky, "honest_auc": honest_auc},
        grading={"trap_keywords": ["leak", "leakage", "target", "deterministic",
                                   "perfect", "exclude", "near-perfect", "f4"]},
    )


def lk_grade(inst, sub):
    got = sub.answers.get("leaky_features") or []
    got = [str(x).strip() for x in got] if isinstance(got, list) else []
    set_ok = set(got) == set(inst.answer["leaky_features"])
    auc_ok = approx(sub.answers.get("honest_auc"), inst.answer["honest_auc"], rtol=0.12)
    return standard_grade(inst, sub, correct_now=(set_ok and auc_ok),
                          trap_keywords=inst.grading["trap_keywords"])


def _lk_solve(view: AgentView, careful: bool) -> Submission:
    df = pd.read_csv(io.StringIO(view.assets["data.csv"]))
    y = df["y"].to_numpy()
    feats = [c for c in df.columns if c != "y"]
    leaky, issues = [], []
    if careful:
        for c in feats:
            if _single_auc(df[c].to_numpy(), y) > 0.985:
                leaky.append(c)
                issues.append(f"feature {c} alone separates y almost perfectly (target leakage); excluded")
    use = [c for c in feats if c not in leaky]
    auc = _cv_auc(df[use].to_numpy(), y)
    return Submission(answers={"leaky_features": leaky, "honest_auc": round(auc, 4)},
                      issues_detected=issues)


# --------------------------------------------------------------------------- #
# Family 2 — Simpson's paradox / confounding.
# --------------------------------------------------------------------------- #
SP = "stats.simpson"


def _simpson_counts(g: np.random.Generator, reversal: bool):
    # (A_rate, B_rate, A_n, B_n) per stratum. Margins keep rounding from flipping.
    if reversal:  # B better in both strata, A better pooled
        strata = [(0.85, 0.90, 200, 50), (0.30, 0.35, 40, 180)]
    else:         # A better in both strata and pooled
        strata = [(0.80, 0.60, 120, 120), (0.50, 0.30, 80, 80)]
    rows = []
    for i, (ar, br, an, bn) in enumerate(strata):
        ar = min(max(ar + g.uniform(-0.02, 0.02), 0.02), 0.98)
        br = min(max(br + g.uniform(-0.02, 0.02), 0.02), 0.98)
        an = int(an * (1 + g.uniform(-0.1, 0.1)))
        bn = int(bn * (1 + g.uniform(-0.1, 0.1)))
        rows.append((f"S{i}", "A", an, int(round(ar * an))))
        rows.append((f"S{i}", "B", bn, int(round(br * bn))))
    return rows


def _simpson_truth(rows):
    df = pd.DataFrame(rows, columns=["stratum", "treatment", "n", "successes"])
    _g = df.groupby("treatment")[["successes", "n"]].sum()
    pooled = _g["successes"] / _g["n"]
    pooled_A_better = pooled["A"] > pooled["B"]
    # within-stratum weighted difference (A - B), weighted by stratum size
    diff, wsum = 0.0, 0.0
    for s, d in df.groupby("stratum"):
        ra = d[d.treatment == "A"].successes.sum() / d[d.treatment == "A"].n.sum()
        rb = d[d.treatment == "B"].successes.sum() / d[d.treatment == "B"].n.sum()
        w = d.n.sum()
        diff += (ra - rb) * w
        wsum += w
    strat_A_better = (diff / wsum) > 0
    return df, bool(pooled_A_better), bool(strat_A_better)


def sp_generate(seed: int, variant: str) -> TaskInstance:
    g = _rng(f"{SP}:{seed}")
    rows = _simpson_counts(g, reversal=(variant == "trapped"))
    df, pooled_A_better, strat_A_better = _simpson_truth(rows)
    trap = pooled_A_better != strat_A_better
    prompt = ("results.csv reports successes out of n for treatments A and B across patient "
              "strata. Does treatment A have a higher success rate than treatment B? Base your "
              "conclusion on the appropriate analysis of these data.")
    return TaskInstance(
        task_id=_tid(SP, seed, variant), family=SP, domain="statistics",
        variant=variant, seed=seed, prompt=prompt, assets={"results.csv": _csv(df)},
        answer_fields={"A_better": "true if treatment A is genuinely better, else false"},
        trap_type=TrapType.CONFOUNDING if trap else None,
        trap_note=("Simpson's paradox: pooled comparison reverses the within-stratum truth" if trap else ""),
        answer={"A_better": strat_A_better},
        grading={"trap_keywords": ["simpson", "confound", "stratif", "pool", "subgroup",
                                   "aggregat", "reverse", "lurking"]},
    )


def sp_grade(inst, sub):
    correct = parse_bool(sub.answers.get("A_better")) == inst.answer["A_better"]
    return standard_grade(inst, sub, correct_now=correct,
                          trap_keywords=inst.grading["trap_keywords"])


def _sp_solve(view: AgentView, careful: bool) -> Submission:
    df = pd.read_csv(io.StringIO(view.assets["results.csv"]))
    _g = df.groupby("treatment")[["successes", "n"]].sum()
    pooled = _g["successes"] / _g["n"]
    pooled_A_better = bool(pooled["A"] > pooled["B"])
    if not careful:
        return Submission(answers={"A_better": pooled_A_better})
    _, _, strat_A_better = _simpson_truth(list(df.itertuples(index=False, name=None)))
    issues = []
    if strat_A_better != pooled_A_better:
        issues.append("Simpson's paradox: pooled trend reverses within strata; stratified analysis used")
    return Submission(answers={"A_better": strat_A_better}, issues_detected=issues)


# --------------------------------------------------------------------------- #
# Family 3 — uncorrected multiple comparisons.
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
    g = _rng(f"{MC}:{seed}")
    if variant == "trapped":
        m, n, real = 40, 60, []
        for _ in range(400):  # ensure >=2 raw hits but none survive BH
            grp, cols, p = _feature_pvals(g, m, n, real)
            if (p < 0.05).sum() >= 2 and not _bh(p).any():
                break
    else:
        m, n, real = 6, 60, [0]
        for _ in range(400):  # ensure only the real feature is a raw hit
            grp, cols, p = _feature_pvals(g, m, n, real)
            raw = set(np.where(p < 0.05)[0])
            if raw == {0} and _bh(p)[0]:
                break
    names = list(cols.keys())
    sig = _bh(p, 0.05)
    df = pd.DataFrame({**cols, "group": grp})
    prompt = (f"data.csv has {m} feature columns f* and a binary group label. Which features are "
              f"genuinely associated with the group? Test each feature and report only those that "
              f"remain significant after appropriate multiple-comparison control at level 0.05.")
    return TaskInstance(
        task_id=_tid(MC, seed, variant), family=MC, domain="statistics",
        variant=variant, seed=seed, prompt=prompt, assets={"data.csv": _csv(df)},
        answer_fields={"significant_features": "list of feature names truly associated (after correction)"},
        trap_type=TrapType.MULTIPLE_COMPARISONS if variant == "trapped" else None,
        trap_note=("several features cross raw p<0.05 by chance; none survive FDR control"
                   if variant == "trapped" else ""),
        answer={"significant_features": [names[i] for i in np.where(sig)[0]]},
        grading={"trap_keywords": ["correct", "bonferroni", "fdr", "benjamini", "multiple compar",
                                   "false discovery", "adjust", "chance"]},
    )


def mc_grade(inst, sub):
    got = sub.answers.get("significant_features") or []
    got = set(str(x).strip() for x in got) if isinstance(got, list) else set()
    correct = got == set(inst.answer["significant_features"])
    return standard_grade(inst, sub, correct_now=correct,
                          trap_keywords=inst.grading["trap_keywords"])


def _mc_solve(view: AgentView, careful: bool) -> Submission:
    df = pd.read_csv(io.StringIO(view.assets["data.csv"]))
    grp = df["group"].to_numpy()
    feats = [c for c in df.columns if c != "group"]
    p = np.array([stats.ttest_ind(df[c][grp == 0], df[c][grp == 1]).pvalue for c in feats])
    if careful:
        sig = _bh(p, 0.05)
        issues = ([f"{int((p < 0.05).sum())} features cross raw p<0.05; applied "
                   f"Benjamini-Hochberg FDR control"] if (p < 0.05).sum() > sum(sig) else [])
        names = [feats[i] for i in np.where(sig)[0]]
    else:
        names = [feats[i] for i in np.where(p < 0.05)[0]]  # uncorrected
        issues = []
    return Submission(answers={"significant_features": names}, issues_detected=issues)


# --------------------------------------------------------------------------- #
# Family 4 — conclusion from an underpowered sample.
# --------------------------------------------------------------------------- #
UP = "stats.underpowered"
N_MIN = 10


def up_generate(seed: int, variant: str) -> TaskInstance:
    g = _rng(f"{UP}:{seed}")
    if variant == "trapped":
        nper = 4
        for _ in range(2000):  # null effect, but this small sample happens to give p<0.05
            a, b = g.normal(size=nper), g.normal(size=nper)
            if stats.ttest_ind(a, b).pvalue < 0.05:
                break
        supports = False  # null effect + underpowered: must not claim a difference
    else:
        nper = 30
        for _ in range(2000):  # adequate n + real effect that the data actually shows
            a, b = g.normal(size=nper), g.normal(loc=0.9, size=nper)
            if stats.ttest_ind(a, b).pvalue < 0.05:
                break
        supports = True

    df = pd.DataFrame({"group": ["A"] * nper + ["B"] * nper,
                       "value": np.concatenate([a, b])})
    prompt = ("measurements.csv has values for groups A and B. Is there real evidence that the "
              "two groups differ (alpha=0.05)? Report whether the evidence supports a genuine "
              "difference, and judge whether the sample is adequate to draw that conclusion.")
    return TaskInstance(
        task_id=_tid(UP, seed, variant), family=UP, domain="statistics",
        variant=variant, seed=seed, prompt=prompt, assets={"measurements.csv": _csv(df)},
        answer_fields={"supports_difference": "true/false",
                       "n_per_group": "sample size per group"},
        trap_type=TrapType.UNDERPOWERED if variant == "trapped" else None,
        trap_note=("n=4/group with no true effect; a chance p<0.05 must not be trusted"
                   if variant == "trapped" else ""),
        answer={"supports_difference": supports},
        grading={"trap_keywords": ["underpower", "sample size", "too small", "power",
                                   "insufficient", "not enough", "n=4", "wide", "unreliable"]},
    )


def up_grade(inst, sub):
    correct = parse_bool(sub.answers.get("supports_difference")) == inst.answer["supports_difference"]
    return standard_grade(inst, sub, correct_now=correct,
                          trap_keywords=inst.grading["trap_keywords"])


def _up_solve(view: AgentView, careful: bool) -> Submission:
    df = pd.read_csv(io.StringIO(view.assets["measurements.csv"]))
    a = df[df.group == "A"].value.to_numpy()
    b = df[df.group == "B"].value.to_numpy()
    nper = min(len(a), len(b))
    p = stats.ttest_ind(a, b).pvalue
    if careful and nper < N_MIN:
        return Submission(answers={"supports_difference": False, "n_per_group": nper},
                          issues_detected=[f"underpowered: n={nper}/group is too small to trust a p-value"])
    return Submission(answers={"supports_difference": bool(p < 0.05), "n_per_group": nper})


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
for fam in [
    Family(LK, "statistics", "Honest AUC with target leakage",
           "Estimate classifier performance; a leaking feature must be found and excluded.",
           [TrapType.DATA_LEAKAGE], lk_generate, lk_grade,
           lambda v: _lk_solve(v, False), lambda v: _lk_solve(v, True)),
    Family(SP, "statistics", "Simpson's paradox",
           "Compare two treatments where pooling reverses the within-stratum truth.",
           [TrapType.CONFOUNDING], sp_generate, sp_grade,
           lambda v: _sp_solve(v, False), lambda v: _sp_solve(v, True)),
    Family(MC, "statistics", "Multiple comparisons",
           "Find truly associated features among many; raw p<0.05 hits arise by chance.",
           [TrapType.MULTIPLE_COMPARISONS], mc_generate, mc_grade,
           lambda v: _mc_solve(v, False), lambda v: _mc_solve(v, True)),
    Family(UP, "statistics", "Underpowered conclusion",
           "Decide if two groups differ when the trapped sample is far too small to trust.",
           [TrapType.UNDERPOWERED], up_generate, up_grade,
           lambda v: _up_solve(v, False), lambda v: _up_solve(v, True)),
]:
    registry.register(fam)
