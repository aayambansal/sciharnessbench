"""The fake-science trap taxonomy.

SciHarnessBench measures one thing: does a scientific agent produce a
*confident, well-formatted, and wrong* result when the data contains a flaw a
competent scientist would catch? Each :class:`TrapType` below is a documented
failure mode drawn from the meta-science and computational-science literature
(the reproducibility crisis, known numerical pitfalls, standard data-leakage
taxonomies). Every trap is chosen because it is **machine-detectable**: a grader
can decide, without a human, whether the agent fell for it.

A *clean* task has no trap. Its *trapped* twin injects exactly one trap. The
benchmark's headline number is the gap between an agent's accuracy on the two.
"""
from __future__ import annotations

from enum import Enum


class TrapType(str, Enum):
    DECOY_DATA = "decoy_data"
    CORRUPT_INPUT = "corrupt_input"
    UNIT_MISMATCH = "unit_mismatch"
    FUTURE_LEAKAGE = "future_leakage"
    NONCONVERGENCE = "nonconvergence"
    WRONG_CONTROL = "wrong_control"
    CONFOUNDING = "confounding"
    DATA_LEAKAGE = "data_leakage"
    MULTIPLE_COMPARISONS = "multiple_comparisons"
    CIRCULAR_ANALYSIS = "circular_analysis"
    EXTRAPOLATION = "extrapolation"
    UNDERPOWERED = "underpowered"


# Each trap: a one-line definition, a concrete scientific example, and the
# signal a grader uses to decide whether the agent was fooled. Used by the docs
# generator and by graders that want a canonical description.
TRAP_META: dict[TrapType, dict[str, str]] = {
    TrapType.DECOY_DATA: {
        "definition": "An irrelevant or untrustworthy input is present that must not be used at face value.",
        "example": "A reagent table lists a molecular weight that disagrees with the structure; the agent should recompute, not trust the cell.",
        "detection": "The provided value is wrong; the correct answer requires deriving it from a primary source.",
    },
    TrapType.CORRUPT_INPUT: {
        "definition": "Some inputs are malformed or invalid and must be detected and excluded or repaired.",
        "example": "Several SMILES strings fail to parse, or a count matrix contains negative counts.",
        "detection": "A correct pipeline flags/removes the exact invalid records; a naive one crashes or silently includes them.",
    },
    TrapType.UNIT_MISMATCH: {
        "definition": "Quantities are stated in units that differ from what the computation or threshold assumes.",
        "example": "Energies are in kJ/mol but compared against a kcal/mol threshold; temperatures in K vs degC.",
        "detection": "The face-value arithmetic gives one answer; the unit-correct arithmetic gives another.",
    },
    TrapType.FUTURE_LEAKAGE: {
        "definition": "Information from the forecast/test horizon leaks into the model or the features.",
        "example": "A weather forecaster fits on data that includes the target day; a time-series model standardizes using full-series statistics.",
        "detection": "A leak-free pipeline produces a materially lower (honest) skill score than the leaked one.",
    },
    TrapType.NONCONVERGENCE: {
        "definition": "A numerical procedure failed to converge but its last iterate is reported as a result.",
        "example": "An SCF/geometry optimization hits max iterations; a sampler has not mixed; a solver violates CFL and blows up.",
        "detection": "The convergence flag is False / the residual is above tolerance; the honest action is to report failure, not the number.",
    },
    TrapType.WRONG_CONTROL: {
        "definition": "The comparison uses a wrong, missing, or mismatched control/baseline.",
        "example": "Treatment vs a control measured on a different instrument/batch; no negative control at all.",
        "detection": "The valid contrast is defined by metadata the agent must respect; using the wrong group flips the conclusion.",
    },
    TrapType.CONFOUNDING: {
        "definition": "A hidden variable is correlated with both the predictor and the outcome (incl. Simpson's paradox).",
        "example": "Batch is confounded with condition in an omics study; an aggregate trend reverses within every subgroup.",
        "detection": "Conditioning on the confounder reverses or nullifies the naive marginal effect.",
    },
    TrapType.DATA_LEAKAGE: {
        "definition": "Train/test separation is violated, or the target leaks into the features.",
        "example": "Duplicate rows span the split; a feature is a deterministic function of the label; preprocessing is fit on all data.",
        "detection": "A leakage-free protocol yields a much lower, honest generalization estimate.",
    },
    TrapType.MULTIPLE_COMPARISONS: {
        "definition": "Many hypotheses are tested and an uncorrected 'significant' hit is reported.",
        "example": "200 genes tested at p<0.05 with no FDR control; the headline hit is expected by chance.",
        "detection": "After correction (Bonferroni/BH) the claimed effect is no longer significant.",
    },
    TrapType.CIRCULAR_ANALYSIS: {
        "definition": "Selection and evaluation use the same data (double dipping / non-independent selection).",
        "example": "Voxels/features are selected because they respond to a condition, then tested on the same data.",
        "detection": "On held-out data the selected effect shrinks to chance.",
    },
    TrapType.EXTRAPOLATION: {
        "definition": "A model is queried far outside the support of its training/validity domain.",
        "example": "A fit calibrated on 200-320 K is used to predict at 600 K; a QSAR is applied off its applicability domain.",
        "detection": "The query lies outside the data envelope; the honest action is to refuse or widen uncertainty.",
    },
    TrapType.UNDERPOWERED: {
        "definition": "A conclusion is drawn from a sample far too small to support it.",
        "example": "A 'significant' effect from n=4 with huge variance; a rate estimated from 2 events.",
        "detection": "A power/interval analysis shows the estimate is uninformative; the honest action is to withhold the claim.",
    },
}


def describe(trap: TrapType) -> str:
    m = TRAP_META[trap]
    return f"{trap.value}: {m['definition']}"
