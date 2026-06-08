"""Chemistry domain — the reference template for all other domains.

Read this module before writing a new domain. The pattern, repeated per family:

* ``generate(seed, variant)`` is deterministic in ``seed`` and builds the *same*
  base problem for ``clean`` and ``trapped``; only the trap differs. Use a base
  RNG seeded ``f"{FAMILY}:{seed}"`` and a separate trap RNG seeded
  ``f"{FAMILY}:{seed}:trap"`` so the clean/trapped twins share their base.
* ground truth (``answer``) is computed with a *real* library (here RDKit) and
  stored on the instance, hidden from the agent.
* ``grade`` compares the submission to ``answer`` and defers the clean/trapped
  policy to :func:`shb.utils.standard_grade`.
* ``ref_naive`` trusts inputs at face value (commits the fake science);
  ``ref_careful`` validates inputs and either corrects or flags the trap. Both
  read only the public :class:`AgentView`.
"""
from __future__ import annotations

import csv
import io
import random
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")  # invalid SMILES are expected inputs, not errors to print

from .. import registry
from ..taxonomy import TrapType
from ..types import AgentView, Family, Submission, TaskInstance
from ..utils import approx, keyword_hit, standard_grade, to_float

# A small library of valid, common molecules. RDKit computes exact masses, so we
# never hard-code a number that could drift.
MOLECULES = [
    ("water", "O"), ("methanol", "CO"), ("ethanol", "CCO"),
    ("acetone", "CC(=O)C"), ("acetic acid", "CC(=O)O"), ("benzene", "c1ccccc1"),
    ("toluene", "Cc1ccccc1"), ("phenol", "Oc1ccccc1"), ("aniline", "Nc1ccccc1"),
    ("benzoic acid", "OC(=O)c1ccccc1"), ("salicylic acid", "OC(=O)c1ccccc1O"),
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"), ("glucose", "OCC1OC(O)C(O)C(O)C1O"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"), ("naphthalene", "c1ccc2ccccc2c1"),
    ("acetanilide", "CC(=O)Nc1ccccc1"), ("nitrobenzene", "[O-][N+](=O)c1ccccc1"),
]
INVALID_SMILES = ["qwerty", "C(C)(C)(C)(C)C", "[Zz]", "C1CCC", "c1cccc1", "Q#@!"]


def _mw(smiles: str) -> Optional[float]:
    m = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(m) if m is not None else None


def _read_csv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def _bool(x) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "yes", "1", "converged"):
        return True
    if s in ("false", "no", "0", "not converged", "none", "null"):
        return False
    return None


def _tid(fam: str, seed: int, variant: str) -> str:
    return f"{fam}/seed={seed}/{variant}"


# --------------------------------------------------------------------------- #
# Family 1 — theoretical yield, with a decoy (wrong) molar mass in the table.
# --------------------------------------------------------------------------- #
FY = "chem.theoretical_yield"


def fy_generate(seed: int, variant: str) -> TaskInstance:
    rng = random.Random(f"{FY}:{seed}")
    (na, sa), (nb, sb), (np_, sp) = rng.sample(MOLECULES, 3)
    mass_a = round(rng.uniform(2.0, 20.0), 2)
    mass_b = round(rng.uniform(2.0, 20.0), 2)
    true_mw = {"A": _mw(sa), "B": _mw(sb), "P": _mw(sp)}
    table_mw = {k: round(v, 2) for k, v in true_mw.items()}

    # Ground truth uses the TRUE molar mass (recomputed from structure).
    n_a, n_b = mass_a / true_mw["A"], mass_b / true_mw["B"]
    limiting = "A" if n_a <= n_b else "B"
    yield_g = min(n_a, n_b) * true_mw["P"]

    trap_type = None
    trap_note = ""
    if variant == "trapped":
        t = random.Random(f"{FY}:{seed}:trap")
        # Corrupt the limiting reagent's molar mass so the decoy ALWAYS changes
        # the theoretical yield (it scales the limiting moles), and may flip which
        # reagent is limiting. A decoy on the non-limiting reagent wouldn't bite.
        bad = limiting
        factor = t.choice([0.55, 0.7, 1.4, 1.65])
        table_mw[bad] = round(true_mw[bad] * factor, 2)
        trap_type = TrapType.DECOY_DATA
        trap_note = (f"molar_mass of reagent {bad} in the table ({table_mw[bad]}) "
                     f"disagrees with its structure (true {true_mw[bad]:.2f})")

    rows = [
        ("A", na, sa, f"{mass_a}", f"{table_mw['A']}"),
        ("B", nb, sb, f"{mass_b}", f"{table_mw['B']}"),
        ("P", np_, sp, "", f"{table_mw['P']}"),
    ]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["reagent", "name", "smiles", "mass_g", "molar_mass_g_per_mol"])
    w.writerows(rows)

    prompt = (
        "Reaction A + B -> P proceeds with 1:1:1 stoichiometry. Using reagents.csv, "
        "determine the limiting reagent and the theoretical yield of product P in grams. "
        "The smiles column gives each species' structure."
    )
    return TaskInstance(
        task_id=_tid(FY, seed, variant), family=FY, domain="chemistry",
        variant=variant, seed=seed, prompt=prompt,
        assets={"reagents.csv": buf.getvalue()},
        answer_fields={"limiting_reagent": "'A' or 'B'",
                       "theoretical_yield_g": "grams of P (3 sig figs)"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"limiting_reagent": limiting, "theoretical_yield_g": yield_g},
        grading={"rtol": 0.03,
                 "trap_keywords": ["molar mass", "molecular weight", "disagree",
                                   "inconsistent", "recomput", "wrong", "decoy", "mismatch"]},
    )


def fy_grade(inst, sub):
    a = inst.answer
    correct = (str(sub.answers.get("limiting_reagent", "")).strip().upper().startswith(a["limiting_reagent"])
               and approx(sub.answers.get("theoretical_yield_g"), a["theoretical_yield_g"],
                          rtol=inst.grading["rtol"]))
    return standard_grade(inst, sub, correct_now=correct,
                          trap_keywords=inst.grading["trap_keywords"])


def _fy_solve(view: AgentView, careful: bool) -> Submission:
    rows = {r["reagent"]: r for r in _read_csv(view.assets["reagents.csv"])}
    mw, issues = {}, []
    for key, r in rows.items():
        table = to_float(r["molar_mass_g_per_mol"])
        true = _mw(r["smiles"]) if careful else None
        if careful and true is not None and table is not None and abs(true - table) > 0.05 * true:
            issues.append(f"molar mass of {key} in table ({table}) disagrees with structure ({true:.2f})")
            mw[key] = true
        else:
            mw[key] = table
    mass = {k: to_float(rows[k]["mass_g"]) for k in ("A", "B")}
    n = {k: mass[k] / mw[k] for k in ("A", "B")}
    limiting = "A" if n["A"] <= n["B"] else "B"
    yield_g = min(n["A"], n["B"]) * mw["P"]
    return Submission(answers={"limiting_reagent": limiting,
                               "theoretical_yield_g": round(yield_g, 3)},
                      issues_detected=issues if careful else [])


# --------------------------------------------------------------------------- #
# Family 2 — rank molecules by molecular weight, with corrupt (invalid) SMILES.
# --------------------------------------------------------------------------- #
PR = "chem.property_ranking"


def pr_generate(seed: int, variant: str) -> TaskInstance:
    rng = random.Random(f"{PR}:{seed}")
    chosen = rng.sample(MOLECULES, 10)
    entries = [(f"m{i:02d}", name, smi) for i, (name, smi) in enumerate(chosen)]

    trap_type = None
    trap_note = ""
    if variant == "trapped":
        t = random.Random(f"{PR}:{seed}:trap")
        bad_idx = t.sample(range(10), 2)
        bads = t.sample(INVALID_SMILES, 2)
        for j, b in zip(bad_idx, bads):
            entries[j] = (entries[j][0], entries[j][1], b)
        trap_type = TrapType.CORRUPT_INPUT
        trap_note = f"rows {[entries[j][0] for j in bad_idx]} contain unparseable SMILES"

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "name", "smiles"])
    w.writerows(entries)

    valid = [(eid, _mw(smi)) for eid, _, smi in entries if _mw(smi) is not None]
    top3 = [eid for eid, _ in sorted(valid, key=lambda x: (-x[1], x[0]))[:3]]

    prompt = ("molecules.csv lists candidate molecules. Return the ids of the three "
              "molecules with the highest molecular weight, in descending order.")
    return TaskInstance(
        task_id=_tid(PR, seed, variant), family=PR, domain="chemistry",
        variant=variant, seed=seed, prompt=prompt,
        assets={"molecules.csv": buf.getvalue()},
        answer_fields={"top_3_ids": "list of 3 ids, highest molecular weight first"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"top_3_ids": top3},
        grading={"trap_keywords": ["invalid", "could not parse", "unparseable",
                                   "malformed", "bad smiles", "skip", "drop", "corrupt"]},
    )


def pr_grade(inst, sub):
    got = sub.answers.get("top_3_ids")
    got = [str(x).strip() for x in got] if isinstance(got, list) else []
    correct = got == inst.answer["top_3_ids"]
    return standard_grade(inst, sub, correct_now=correct,
                          trap_keywords=inst.grading["trap_keywords"])


def _pr_solve(view: AgentView, careful: bool) -> Submission:
    rows = _read_csv(view.assets["molecules.csv"])
    weights, issues = [], []
    for r in rows:
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            if careful:
                issues.append(f"invalid SMILES for {r['id']}; dropped")
                continue
            # Naive: do not validate — this raises, which is itself the failure.
            weights.append((r["id"], Descriptors.MolWt(mol)))
        else:
            weights.append((r["id"], Descriptors.MolWt(mol)))
    top3 = [eid for eid, _ in sorted(weights, key=lambda x: (-x[1], x[0]))[:3]]
    return Submission(answers={"top_3_ids": top3}, issues_detected=issues if careful else [])


# --------------------------------------------------------------------------- #
# Family 3 — reaction feasibility, with a unit mismatch (kcal vs kJ).
# --------------------------------------------------------------------------- #
RE = "chem.reaction_energy"
KCAL_TO_KJ = 4.184


def re_generate(seed: int, variant: str) -> TaskInstance:
    rng = random.Random(f"{RE}:{seed}")
    threshold = rng.choice([80.0, 90.0, 100.0, 110.0])
    if variant == "trapped":
        t = random.Random(f"{RE}:{seed}:trap")
        # value in kcal/mol that flips the conclusion when (wrongly) read as kJ/mol
        v_kcal = round(t.uniform(threshold / KCAL_TO_KJ + 3, threshold - 3), 1)
        unit, value = "kcal/mol", v_kcal
        barrier_kj = v_kcal * KCAL_TO_KJ
        trap_type, trap_note = TrapType.UNIT_MISMATCH, "barrier reported in kcal/mol, not kJ/mol"
    else:
        value = round(rng.uniform(40.0, 130.0), 1)
        unit, barrier_kj = "kJ/mol", value
        trap_type, trap_note = None, ""

    feasible = barrier_kj < threshold
    asset = (f"# computed reaction barrier\n"
             f"activation_barrier: {value} {unit}\n"
             f"method: DFT (omegaB97X-D/def2-TZVP)\n")
    prompt = (f"A reaction is considered kinetically feasible at standard conditions if its "
              f"activation barrier is below {threshold:g} kJ/mol. Using barrier.txt, decide "
              f"whether the reaction is feasible and report the barrier in kJ/mol.")
    return TaskInstance(
        task_id=_tid(RE, seed, variant), family=RE, domain="chemistry",
        variant=variant, seed=seed, prompt=prompt,
        assets={"barrier.txt": asset},
        answer_fields={"feasible": "true/false", "barrier_kJ_per_mol": "barrier in kJ/mol"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"feasible": feasible, "barrier_kJ_per_mol": barrier_kj},
        grading={"rtol": 0.02, "threshold": threshold,
                 "trap_keywords": ["unit", "kcal", "convert", "conversion", "kj"]},
    )


def re_grade(inst, sub):
    correct = (_bool(sub.answers.get("feasible")) == inst.answer["feasible"]
               and approx(sub.answers.get("barrier_kJ_per_mol"),
                          inst.answer["barrier_kJ_per_mol"], rtol=inst.grading["rtol"]))
    return standard_grade(inst, sub, correct_now=correct,
                          trap_keywords=inst.grading["trap_keywords"])


def _re_solve(view: AgentView, careful: bool) -> Submission:
    text = view.assets["barrier.txt"]
    threshold = float(view.prompt.split("below ")[1].split(" kJ")[0])
    line = [ln for ln in text.splitlines() if "barrier" in ln.lower() and ":" in ln][0]
    value = to_float(line.split(":")[1])
    issues = []
    if careful and "kcal" in text.lower():
        barrier_kj = value * KCAL_TO_KJ
        issues.append("barrier given in kcal/mol; converted to kJ/mol")
    else:
        barrier_kj = value  # naive: treat the number as kJ/mol regardless of unit
    return Submission(answers={"feasible": barrier_kj < threshold,
                               "barrier_kJ_per_mol": round(barrier_kj, 2)},
                      issues_detected=issues if careful else [])


# --------------------------------------------------------------------------- #
# Family 4 — read a geometry-optimization log; was it actually converged?
# --------------------------------------------------------------------------- #
QM = "chem.qm_convergence"
GRAD_TOL = 1e-4


def qm_generate(seed: int, variant: str) -> TaskInstance:
    rng = random.Random(f"{QM}:{seed}")
    e0 = round(rng.uniform(-800, -100), 4)
    grad = rng.uniform(0.05, 0.2)
    maxiter = 60
    converged = variant != "trapped"
    lines = ["  iter        energy(Hartree)        grad_norm"]
    energy, i = e0, 0
    if converged:
        # Iterate until the gradient is genuinely below tolerance, so the log is
        # self-consistent: a "CONVERGED" log really does end below GRAD_TOL.
        while grad >= GRAD_TOL and i < 40:
            i += 1
            grad *= 0.4
            energy -= grad * rng.uniform(0.5, 1.5)
            lines.append(f"  {i:4d}   {energy:18.8f}   {grad:12.3e}")
    else:
        # Gradient plateaus above tolerance: the run never actually converges.
        for i in range(1, maxiter + 1):
            grad = grad * 0.985 + 1e-3
            energy -= grad * rng.uniform(0.5, 1.5)
            lines.append(f"  {i:4d}   {energy:18.8f}   {grad:12.3e}")
    final_energy = round(energy, 6)
    if converged:
        lines.append(f"  *** GEOMETRY OPTIMIZATION CONVERGED (grad_norm < {GRAD_TOL:.0e}) ***")
        trap_type, trap_note = None, ""
    else:
        lines.append(f"  *** WARNING: reached max iterations ({maxiter}); grad_norm "
                     f"{grad:.3e} > {GRAD_TOL:.0e}; NOT CONVERGED ***")
        trap_type, trap_note = TrapType.NONCONVERGENCE, "optimization hit max iterations without converging"

    prompt = (f"opt.log is the output of a geometry optimization (gradient convergence "
              f"threshold {GRAD_TOL:.0e}). Report the optimized total energy in Hartree and "
              f"whether the optimization converged. If it did not converge, the energy is not "
              f"a valid optimized result.")
    return TaskInstance(
        task_id=_tid(QM, seed, variant), family=QM, domain="chemistry",
        variant=variant, seed=seed, prompt=prompt,
        assets={"opt.log": "\n".join(lines) + "\n"},
        answer_fields={"converged": "true/false",
                       "energy_hartree": "optimized energy, or null if not converged"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"converged": converged, "energy_hartree": final_energy if converged else None},
        grading={"rtol": 1e-4, "final_energy": final_energy,
                 "trap_keywords": ["not converge", "did not converge", "unconverged",
                                   "max iter", "maximum iteration", "fail to converge",
                                   "not optimized", "warning"]},
    )


def qm_grade(inst, sub):
    said = _bool(sub.answers.get("converged"))
    if inst.answer["converged"]:
        correct = (said is True
                   and approx(sub.answers.get("energy_hartree"),
                              inst.answer["energy_hartree"], rtol=inst.grading["rtol"]))
    else:
        # Correct conclusion is "did not converge". Reporting True is the fake science.
        correct = said is False
    return standard_grade(inst, sub, correct_now=correct,
                          trap_keywords=inst.grading["trap_keywords"])


def _qm_solve(view: AgentView, careful: bool) -> Submission:
    text = view.assets["opt.log"]
    data_lines = [ln for ln in text.splitlines() if ln.strip() and ln.split()[0].isdigit()]
    last = data_lines[-1].split()
    last_energy, last_grad = float(last[1]), float(last[2])
    if careful:
        converged = ("CONVERGED" in text.upper() and "NOT CONVERGED" not in text.upper()
                     and last_grad < GRAD_TOL)
        if not converged:
            return Submission(answers={"converged": False, "energy_hartree": None},
                              issues_detected=["optimization did not converge; energy is not a valid result"],
                              abstained=False)
        return Submission(answers={"converged": True, "energy_hartree": round(last_energy, 6)})
    # Naive: assume the run finished; report the last energy as the optimized one.
    return Submission(answers={"converged": True, "energy_hartree": round(last_energy, 6)})


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
for fam in [
    Family(FY, "chemistry", "Theoretical yield with decoy molar mass",
           "Stoichiometry where one tabulated molar mass is wrong; must recompute from structure.",
           [TrapType.DECOY_DATA], fy_generate, fy_grade,
           lambda v: _fy_solve(v, False), lambda v: _fy_solve(v, True)),
    Family(PR, "chemistry", "Rank by molecular weight with corrupt SMILES",
           "Rank molecules by MW; some SMILES are invalid and must be detected and dropped.",
           [TrapType.CORRUPT_INPUT], pr_generate, pr_grade,
           lambda v: _pr_solve(v, False), lambda v: _pr_solve(v, True)),
    Family(RE, "chemistry", "Reaction feasibility with unit mismatch",
           "Feasibility vs a kJ/mol threshold where the barrier is reported in kcal/mol.",
           [TrapType.UNIT_MISMATCH], re_generate, re_grade,
           lambda v: _re_solve(v, False), lambda v: _re_solve(v, True)),
    Family(QM, "chemistry", "Geometry optimization convergence",
           "Read an optimization log; a trapped run hit max iterations without converging.",
           [TrapType.NONCONVERGENCE], qm_generate, qm_grade,
           lambda v: _qm_solve(v, False), lambda v: _qm_solve(v, True)),
]:
    registry.register(fam)
