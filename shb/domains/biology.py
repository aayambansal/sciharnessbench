"""Biology / genomics domain — omics traps a competent analyst would catch.

The three families mirror the chemistry template exactly:

* ``generate(seed, variant)`` is deterministic in ``seed`` and builds the *same*
  base problem for ``clean`` and ``trapped``; only the injected trap differs. A
  base RNG seeded ``f"{FAMILY}:{seed}"`` draws the biology; a separate trap RNG
  seeded ``f"{FAMILY}:{seed}:trap"`` injects the flaw, so the clean/trapped twins
  share their base and the gap isolates the trap.
* ground truth (``answer``) is computed with real libraries (numpy/scipy and
  biopython ``Bio`` where natural) and stored on the instance, hidden.
* ``grade`` compares the submission to ``answer`` and defers the clean/trapped
  policy to :func:`shb.utils.standard_grade`.
* ``ref_naive`` trusts inputs at face value (commits the fake science);
  ``ref_careful`` validates inputs and either corrects or flags the trap. Both
  read only the public :class:`AgentView`.

The traps, one distinct :class:`TrapType` per family:

* CONFOUNDING — an RNA-seq count/expression matrix where, in the trapped twin,
  condition is perfectly aliased with sequencing batch. Naive differential
  expression reports a hugely inflated, batch-driven gene count; the condition
  effect is in fact *not identifiable* and the honest action is to say so.
* WRONG_CONTROL — a treatment group and two candidate controls. The obvious
  (metadata ``role == control``) group is, in the trapped twin, a *different
  tissue* than the treatment; the genuinely matched control is the other group.
  Using the obvious-but-mismatched control flips the fold-change sign.
* CORRUPT_INPUT — DNA sequences as FASTA; report mean GC content. The trapped
  twin slips in invalid records (non-ACGT, empty, long ambiguous ``N`` runs)
  that must be detected and excluded; ground truth uses valid sequences only.
"""
from __future__ import annotations

import hashlib
import io
import logging
import random
from typing import Optional

import numpy as np
from scipy import stats

from Bio import SeqIO, BiopythonWarning
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction

# Invalid/odd FASTA records are expected inputs, not errors to print.
logging.getLogger("Bio").setLevel(logging.CRITICAL)
import warnings as _warnings
_warnings.filterwarnings("ignore", category=BiopythonWarning)

from .. import registry
from ..taxonomy import TrapType
from ..types import AgentView, Family, Submission, TaskInstance
from ..utils import approx, keyword_hit, parse_bool, standard_grade, to_float


def _rng(seed_str: str) -> np.random.Generator:
    # Hash the full key so distinct seeds give distinct problems (a plain
    # little-endian byte read would only see the low bytes and collapse seeds).
    digest = hashlib.sha256(seed_str.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _bh(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR control; returns a boolean significance mask."""
    p = np.asarray(pvals, float)
    m = len(p)
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


# --------------------------------------------------------------------------- #
# Family 1 — RNA-seq differential expression with condition/batch confounding.
# --------------------------------------------------------------------------- #
DE = "bio.diffexpr_confounding"
N_GENES = 60
N_PER = 6                     # samples per condition (treated / control)
DE_Q = 0.05


def _de_matrix(seed: int, confounded: bool):
    """Build a (genes x samples) log-expression matrix plus sample metadata.

    The base biology (baselines, the small true condition signal, the broad
    batch signal, sample order) is fixed by the base RNG, so clean and trapped
    twins are identical except for how condition maps onto batch.
    """
    g = _rng(f"{DE}:{seed}")
    base = g.uniform(4.0, 9.0, size=N_GENES)

    # A small, genuine condition effect on a handful of genes.
    cond_genes = np.sort(g.choice(N_GENES, size=5, replace=False))
    cond_eff = np.zeros(N_GENES)
    cond_eff[cond_genes] = g.choice([-1.0, 1.0], size=5) * g.uniform(1.6, 2.6, size=5)

    # A broad batch effect on many genes (the lurking technical variable).
    batch_genes = g.choice(N_GENES, size=32, replace=False)
    batch_eff = np.zeros(N_GENES)
    batch_eff[batch_genes] = g.choice([-1.0, 1.0], size=32) * g.uniform(1.8, 3.2, size=32)

    conditions = ["treated"] * N_PER + ["control"] * N_PER
    if confounded:
        # Perfectly aliased: every treated sample in batch1, every control in batch2.
        batches = [1] * N_PER + [2] * N_PER
    else:
        # Balanced: each condition split evenly across both batches (identifiable).
        half = N_PER // 2
        batches = ([1] * half + [2] * (N_PER - half)) + ([1] * half + [2] * (N_PER - half))

    # Per-sample expression noise drawn from the base RNG (shared by both twins).
    noise = g.normal(0.0, 0.30, size=(N_GENES, len(conditions)))
    expr = np.zeros((N_GENES, len(conditions)))
    for j, (c, b) in enumerate(zip(conditions, batches)):
        mu = base.copy()
        if c == "treated":
            mu = mu + cond_eff
        if b == 2:
            mu = mu + batch_eff
        expr[:, j] = mu + noise[:, j]
    return expr, conditions, batches, cond_genes


def _de_count(expr: np.ndarray, conditions: list[str]) -> int:
    """BH-significant gene count for a straight treated-vs-control contrast.

    On a *balanced* design this is a valid estimate of the condition effect; on
    a confounded design it is dominated by batch and must not be trusted.
    """
    cond = np.asarray(conditions)
    treated = expr[:, cond == "treated"]
    control = expr[:, cond == "control"]
    p = np.array([stats.ttest_ind(treated[i], control[i]).pvalue
                  for i in range(expr.shape[0])])
    return int(_bh(p, DE_Q).sum())


def de_generate(seed: int, variant: str) -> TaskInstance:
    confounded = variant == "trapped"
    expr, conditions, batches, _ = _de_matrix(seed, confounded)

    # On the clean (balanced) twin a plain contrast is valid -> that count is the
    # ground truth. The careful and naive pipelines compute it identically there.
    clean_count = _de_count(*_de_matrix(seed, False)[:2])

    if confounded:
        trap_type = TrapType.CONFOUNDING
        trap_note = ("condition is perfectly confounded with sequencing batch "
                     "(all treated in batch1, all control in batch2); the "
                     "condition effect is not identifiable")
        identifiable = False
        n_diff = None
    else:
        trap_type = None
        trap_note = ""
        identifiable = True
        n_diff = clean_count

    # counts.csv: gene rows, sample columns.
    sample_ids = [f"s{j+1:02d}" for j in range(len(conditions))]
    header = "gene," + ",".join(sample_ids)
    lines = [header]
    for i in range(expr.shape[0]):
        vals = ",".join(f"{expr[i, j]:.4f}" for j in range(expr.shape[1]))
        lines.append(f"g{i:02d},{vals}")
    counts_csv = "\n".join(lines) + "\n"

    # metadata.csv: one row per sample.
    meta = ["sample,condition,batch"]
    for sid, c, b in zip(sample_ids, conditions, batches):
        meta.append(f"{sid},{c},batch{b}")
    meta_csv = "\n".join(meta) + "\n"

    prompt = (
        "counts.csv is a log-expression matrix (genes in rows, samples in columns) "
        "and metadata.csv gives each sample's condition (treated/control) and "
        "sequencing batch. Report whether the condition effect is identifiable from "
        "this design, and if so the number of genes differentially expressed between "
        "conditions at BH-FDR 0.05. If the design does not allow the condition effect "
        "to be separated from a technical factor, it is not identifiable."
    )
    return TaskInstance(
        task_id=_tid(DE, seed, variant), family=DE, domain="biology",
        variant=variant, seed=seed, prompt=prompt,
        assets={"counts.csv": counts_csv, "metadata.csv": meta_csv},
        answer_fields={
            "identifiable": "true if the condition effect can be separated from batch, else false",
            "n_diff_genes": "number of DE genes at BH-FDR 0.05, or null if not identifiable",
        },
        trap_type=trap_type, trap_note=trap_note,
        answer={"identifiable": identifiable, "n_diff_genes": n_diff},
        grading={"trap_keywords": ["confound", "batch", "not identifiable", "aliased",
                                   "perfectly predict", "perfectly correlat", "nested",
                                   "cannot separate", "can't separate", "indistinguishable"]},
    )


def de_grade(inst, sub):
    a = inst.answer
    said_ident = parse_bool(sub.answers.get("identifiable"))
    if a["identifiable"]:
        n_ok = approx(sub.answers.get("n_diff_genes"), a["n_diff_genes"], rtol=0.0, atol=1.0)
        correct = (said_ident is True) and n_ok
    else:
        # Correct conclusion is "not identifiable". Reporting an inflated count is fake science.
        correct = said_ident is False
    return standard_grade(inst, sub, correct_now=correct,
                          trap_keywords=inst.grading["trap_keywords"])


def _de_read(view: AgentView):
    rows = [ln for ln in view.assets["counts.csv"].splitlines() if ln.strip()]
    sample_ids = rows[0].split(",")[1:]
    expr = np.array([[float(x) for x in ln.split(",")[1:]] for ln in rows[1:]])
    meta = {}
    mlines = [ln for ln in view.assets["metadata.csv"].splitlines() if ln.strip()][1:]
    for ln in mlines:
        sid, cond, batch = ln.split(",")
        meta[sid] = (cond, batch)
    conditions = [meta[s][0] for s in sample_ids]
    batches = [meta[s][1] for s in sample_ids]
    return expr, conditions, batches


def _de_solve(view: AgentView, careful: bool) -> Submission:
    expr, conditions, batches = _de_read(view)
    if careful:
        # Is condition perfectly predicted by batch? (each batch maps to one condition)
        by_batch: dict[str, set] = {}
        for c, b in zip(conditions, batches):
            by_batch.setdefault(b, set()).add(c)
        confounded = all(len(v) == 1 for v in by_batch.values()) and len(by_batch) > 1
        if confounded:
            return Submission(
                answers={"identifiable": False, "n_diff_genes": None},
                issues_detected=["condition is perfectly confounded with batch "
                                 "(batch perfectly predicts condition); the condition "
                                 "effect is not identifiable"],
            )
        n = _de_count(expr, conditions)
        return Submission(answers={"identifiable": True, "n_diff_genes": n})
    # Naive: ignore batch entirely; run condition-vs-condition and report the count.
    n = _de_count(expr, conditions)
    return Submission(answers={"identifiable": True, "n_diff_genes": n})


# --------------------------------------------------------------------------- #
# Family 2 — treatment vs control where the obvious control is the wrong tissue.
# --------------------------------------------------------------------------- #
WC = "bio.wrong_control"
N_REP = 4


def _wc_data(seed: int):
    """Treatment (liver) plus a tissue-matched (liver) and a mismatched (kidney)
    candidate control, for one target gene. The kidney baseline is offset so that
    using it as the control flips the sign of the log2 fold-change.
    """
    g = _rng(f"{WC}:{seed}")
    liver_base = g.uniform(7.0, 9.0)
    direction = float(g.choice([-1.0, 1.0]))
    true_eff = direction * g.uniform(1.1, 1.7)
    gap = g.uniform(1.6, 2.6)
    kidney_base = liver_base + true_eff + direction * gap  # forces wrong-control sign flip

    treat = liver_base + true_eff + g.normal(0.0, 0.15, size=N_REP)
    liver_ctrl = liver_base + g.normal(0.0, 0.15, size=N_REP)
    kidney_ctrl = kidney_base + g.normal(0.0, 0.15, size=N_REP)
    return treat, liver_ctrl, kidney_ctrl


def wc_generate(seed: int, variant: str) -> TaskInstance:
    treat, liver_ctrl, kidney_ctrl = _wc_data(seed)

    # The matched control (same tissue as treatment) is always the liver group.
    # CLEAN: the obvious (role=control) group is the matched liver group.
    # TRAPPED: the obvious (role=control) group is the mismatched kidney group;
    # the matched control is relabeled role=reference. Only the role labels move.
    if variant == "trapped":
        # obvious "control" is kidney (wrong tissue); matched is liver, labeled reference.
        groups = [
            ("treatment", "liver", "treatment", treat),
            ("ctrlA", "kidney", "control", kidney_ctrl),
            ("ctrlB", "liver", "reference", liver_ctrl),
        ]
        trap_type = TrapType.WRONG_CONTROL
        trap_note = ("the group labeled role=control (ctrlA) is kidney tissue, mismatched "
                     "to the liver treatment; the matched control is the liver group (ctrlB)")
    else:
        # obvious "control" is the matched liver group.
        groups = [
            ("treatment", "liver", "treatment", treat),
            ("ctrlA", "liver", "control", liver_ctrl),
            ("ctrlB", "kidney", "reference", kidney_ctrl),
        ]
        trap_type = None
        trap_note = ""

    # expression.csv: one row per (group, replicate).
    lines = ["group,replicate,expression"]
    for gid, _tissue, _role, vals in groups:
        for r, v in enumerate(vals):
            lines.append(f"{gid},r{r+1},{v:.4f}")
    expr_csv = "\n".join(lines) + "\n"

    # samples.csv: metadata mapping group -> tissue, role.
    meta = ["group,tissue,role"]
    for gid, tissue, role, _vals in groups:
        meta.append(f"{gid},{tissue},{role}")
    meta_csv = "\n".join(meta) + "\n"

    # Ground truth uses the tissue-matched control (same tissue as the treatment).
    treat_mean = float(treat.mean())
    matched_mean = float(liver_ctrl.mean())
    log2fc = treat_mean - matched_mean
    upregulated = log2fc > 0

    prompt = (
        "expression.csv gives log2 expression of a target gene for a treatment group and "
        "two candidate control groups; samples.csv gives each group's tissue and role. "
        "Compute the log2 fold-change of the treatment relative to the appropriate control "
        "and state whether the gene is up-regulated (fold-change > 0) by the treatment. "
        "The valid control must match the treatment's tissue."
    )
    return TaskInstance(
        task_id=_tid(WC, seed, variant), family=WC, domain="biology",
        variant=variant, seed=seed, prompt=prompt,
        assets={"expression.csv": expr_csv, "samples.csv": meta_csv},
        answer_fields={
            "log2_fold_change": "treatment minus the tissue-matched control mean",
            "upregulated": "true if the matched-control fold-change is > 0, else false",
        },
        trap_type=trap_type, trap_note=trap_note,
        answer={"log2_fold_change": log2fc, "upregulated": bool(upregulated)},
        grading={"rtol": 0.06,
                 "trap_keywords": ["tissue", "mismatch", "wrong control", "matched control",
                                   "kidney", "liver", "different tissue", "not a valid control",
                                   "role", "matched", "mismatched"]},
    )


def wc_grade(inst, sub):
    a = inst.answer
    fc_ok = approx(sub.answers.get("log2_fold_change"), a["log2_fold_change"], rtol=inst.grading["rtol"])
    up_ok = parse_bool(sub.answers.get("upregulated")) == a["upregulated"]
    return standard_grade(inst, sub, correct_now=(fc_ok and up_ok),
                          trap_keywords=inst.grading["trap_keywords"])


def _wc_read(view: AgentView):
    expr: dict[str, list[float]] = {}
    elines = [ln for ln in view.assets["expression.csv"].splitlines() if ln.strip()][1:]
    for ln in elines:
        gid, _rep, val = ln.split(",")
        expr.setdefault(gid, []).append(float(val))
    meta: dict[str, dict[str, str]] = {}
    mlines = [ln for ln in view.assets["samples.csv"].splitlines() if ln.strip()][1:]
    for ln in mlines:
        gid, tissue, role = ln.split(",")
        meta[gid] = {"tissue": tissue, "role": role}
    return expr, meta


def _wc_solve(view: AgentView, careful: bool) -> Submission:
    expr, meta = _wc_read(view)
    treat_id = next(g for g, m in meta.items() if m["role"] == "treatment")
    treat_mean = float(np.mean(expr[treat_id]))
    treat_tissue = meta[treat_id]["tissue"]
    issues = []
    if careful:
        # Use the control whose tissue matches the treatment; flag if the obvious one doesn't.
        obvious = next((g for g, m in meta.items() if m["role"] == "control"), None)
        matched = next((g for g, m in meta.items()
                        if g != treat_id and m["tissue"] == treat_tissue), None)
        if matched is None:
            return Submission(answers={"log2_fold_change": None, "upregulated": None},
                              issues_detected=["no control matches the treatment tissue"],
                              abstained=True)
        if obvious is not None and meta[obvious]["tissue"] != treat_tissue:
            issues.append(f"the role=control group ({obvious}) is {meta[obvious]['tissue']} tissue, "
                          f"mismatched to the {treat_tissue} treatment; using the tissue-matched "
                          f"control ({matched}) instead")
        ctrl_id = matched
    else:
        # Naive: use whichever group is labeled the control, ignore tissue.
        ctrl_id = next(g for g, m in meta.items() if m["role"] == "control")
    fc = treat_mean - float(np.mean(expr[ctrl_id]))
    return Submission(answers={"log2_fold_change": round(fc, 4), "upregulated": bool(fc > 0)},
                      issues_detected=issues)


# --------------------------------------------------------------------------- #
# Family 3 — mean GC content of DNA sequences, with corrupt FASTA records.
# --------------------------------------------------------------------------- #
GC = "bio.gc_content"
N_SEQS = 12
SEQ_LEN = (60, 120)


def _rand_dna(g: np.random.Generator, length: int) -> str:
    return "".join(g.choice(list("ACGT"), size=length))


def _is_valid_dna(seq: str) -> bool:
    """A usable DNA record: non-empty and composed only of A/C/G/T (any case)."""
    s = seq.strip().upper()
    return len(s) > 0 and set(s) <= set("ACGT")


def gc_generate(seed: int, variant: str) -> TaskInstance:
    g = _rng(f"{GC}:{seed}")
    lengths = [int(g.integers(SEQ_LEN[0], SEQ_LEN[1] + 1)) for _ in range(N_SEQS)]
    seqs = [_rand_dna(g, n) for n in lengths]
    records = [(f"seq{i+1:02d}", s) for i, s in enumerate(seqs)]

    bad_ids: list[str] = []
    if variant == "trapped":
        t = random.Random(f"{GC}:{seed}:trap")
        idx = t.sample(range(N_SEQS), 3)
        # Three distinct corruptions: non-ACGT letters, an empty record, a long N run.
        kinds = ["nonacgt", "empty", "nrun"]
        t.shuffle(kinds)
        for j, kind in zip(idx, kinds):
            sid, orig = records[j]
            if kind == "nonacgt":
                # splice junk (protein/IUPAC garbage) into the middle
                pos = len(orig) // 2
                corrupted = orig[:pos] + "XQ-Z*" + orig[pos:]
            elif kind == "empty":
                corrupted = ""
            else:  # long ambiguous N run dominating the record
                corrupted = "N" * (len(orig) + 5)
            records[j] = (sid, corrupted)
            bad_ids.append(sid)
        bad_ids.sort()
        trap_type = TrapType.CORRUPT_INPUT
        trap_note = (f"records {bad_ids} are invalid (non-ACGT characters, empty, or an "
                     f"all-N run) and must be excluded")
    else:
        trap_type = None
        trap_note = ""

    # FASTA text (wrap at 70 cols, the usual convention).
    def _fmt(sid: str, s: str) -> str:
        body = "\n".join(s[i:i + 70] for i in range(0, len(s), 70)) if s else ""
        return f">{sid}\n{body}\n" if body else f">{sid}\n\n"
    fasta = "".join(_fmt(sid, s) for sid, s in records)

    # Ground truth: mean GC fraction over VALID sequences only, via biopython.
    valid_seqs = [s for _sid, s in records if _is_valid_dna(s)]
    gc_vals = [gc_fraction(Seq(s.upper())) for s in valid_seqs]  # pure ACGT => exact (G+C)/len
    mean_gc = float(np.mean(gc_vals))
    n_valid = len(valid_seqs)

    prompt = (
        "sequences.fasta contains DNA sequences. Report the mean GC content (as a fraction "
        "in [0,1]) and the number of valid sequences used. A valid sequence is non-empty and "
        "contains only the bases A, C, G, T; any record with other characters, an empty body, "
        "or that is entirely ambiguous must be excluded from the calculation."
    )
    return TaskInstance(
        task_id=_tid(GC, seed, variant), family=GC, domain="biology",
        variant=variant, seed=seed, prompt=prompt,
        assets={"sequences.fasta": fasta},
        answer_fields={
            "mean_gc_content": "mean GC fraction over valid sequences (in [0,1])",
            "n_valid_sequences": "count of sequences included after validation",
        },
        trap_type=trap_type, trap_note=trap_note,
        answer={"mean_gc_content": mean_gc, "n_valid_sequences": n_valid},
        grading={"rtol": 0.01,
                 "trap_keywords": ["invalid", "non-acgt", "non acgt", "corrupt", "empty",
                                   "ambiguous", "excluded", "skip", "drop", "bad sequence",
                                   "malformed", "all-n", "all n", "n run"]},
    )


def gc_grade(inst, sub):
    a = inst.answer
    gc_ok = approx(sub.answers.get("mean_gc_content"), a["mean_gc_content"], rtol=inst.grading["rtol"])
    n_ok = approx(sub.answers.get("n_valid_sequences"), a["n_valid_sequences"], rtol=0.0, atol=0.0)
    return standard_grade(inst, sub, correct_now=(gc_ok and n_ok),
                          trap_keywords=inst.grading["trap_keywords"])


def _gc_solve(view: AgentView, careful: bool) -> Submission:
    records = list(SeqIO.parse(io.StringIO(view.assets["sequences.fasta"]), "fasta"))
    gc_vals, issues, n_used = [], [], 0
    for rec in records:
        seq = str(rec.seq)
        if careful:
            if not _is_valid_dna(seq):
                issues.append(f"sequence {rec.id} is invalid (non-ACGT, empty, or ambiguous); excluded")
                continue
            gc_vals.append(gc_fraction(Seq(seq.upper())))
            n_used += 1
        else:
            # Naive: trust every record. gc_fraction over junk/empty silently corrupts the mean.
            gc_vals.append(gc_fraction(rec.seq))
            n_used += 1
    mean_gc = float(np.mean(gc_vals)) if gc_vals else 0.0
    return Submission(answers={"mean_gc_content": round(mean_gc, 6), "n_valid_sequences": n_used},
                      issues_detected=issues if careful else [])


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
for fam in [
    Family(DE, "biology", "Differential expression with batch confounding",
           "Count DE genes between conditions; a trapped design aliases condition with "
           "sequencing batch, so the condition effect is not identifiable.",
           [TrapType.CONFOUNDING], de_generate, de_grade,
           lambda v: _de_solve(v, False), lambda v: _de_solve(v, True)),
    Family(WC, "biology", "Treatment vs the tissue-matched control",
           "Compute a fold-change vs the appropriate control; the obvious control is a "
           "mismatched tissue in the trapped twin and flips the fold-change sign.",
           [TrapType.WRONG_CONTROL], wc_generate, wc_grade,
           lambda v: _wc_solve(v, False), lambda v: _wc_solve(v, True)),
    Family(GC, "biology", "Mean GC content with corrupt FASTA records",
           "Mean GC content of DNA sequences where some FASTA records are invalid and "
           "must be detected and excluded.",
           [TrapType.CORRUPT_INPUT], gc_generate, gc_grade,
           lambda v: _gc_solve(v, False), lambda v: _gc_solve(v, True)),
]:
    registry.register(fam)
