"""Chemistry domain — the reference template for the v2 contract.

Read this module before writing or migrating a domain. Per family:

* ``generate(seed, variant)`` is deterministic and **salted** (``family_rng`` /
  ``np_seed``). The clean and trapped instances for a seed are genuine
  counterfactual twins — identical except the injected flaw — and store a shared
  ``base_signature`` so a test can verify it. Families that cannot be true twins
  set ``paired=False`` (none here; all four are twins).
* ground truth is computed with RDKit and stored in ``answer`` (hidden).
* ``grade`` builds an ``evidence_ok`` check from the hidden ``grading`` payload
  and calls :func:`shb.utils.standard_grade` with the family's ``flaw_kind``.
* ``ref_naive`` trusts inputs and answers confidently (commits fake science);
  ``ref_careful`` validates, corrects, and emits a **structured, evidence-bearing**
  issue. Both read only the public :class:`AgentView`.
* each family supplies a neutral default (cued) prompt and an ``prompt_uncued``
  variant with no trap vocabulary, for the uncued track.
"""
from __future__ import annotations

import csv
import io
import re
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from .. import registry
from ..taxonomy import TrapType
from ..types import AgentView, Family, Submission, TaskInstance
from ..utils import (approx, base_signature, ev_contains, ev_near, ev_text,
                     family_rng, issue, parse_bool, standard_grade, to_float)

RDLogger.DisableLog("rdApp.*")  # invalid SMILES are expected inputs, not errors to print

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


def _tid(fam, seed, variant):
    return f"{fam}/seed={seed}/{variant}"


# --------------------------------------------------------------------------- #
# Family 1 — theoretical yield; decoy molar mass on the limiting reagent.
# --------------------------------------------------------------------------- #
FY = "chem.theoretical_yield"


def fy_generate(seed, variant):
    rng = family_rng(FY, seed)
    (na, sa), (nb, sb), (np_, sp) = rng.sample(MOLECULES, 3)
    mass_a, mass_b = round(rng.uniform(2, 20), 2), round(rng.uniform(2, 20), 2)
    mw = {"A": _mw(sa), "B": _mw(sb), "P": _mw(sp)}
    n_a, n_b = mass_a / mw["A"], mass_b / mw["B"]
    limiting = "A" if n_a <= n_b else "B"
    yield_g = min(n_a, n_b) * mw["P"]
    table_mw = {k: round(v, 2) for k, v in mw.items()}

    bad, trap_type, trap_note = None, None, ""
    if variant == "trapped":
        t = family_rng(FY, seed, trap=True)
        bad = limiting                                  # corrupt the limiting reagent: always bites
        table_mw[bad] = round(mw[bad] * t.choice([0.55, 0.7, 1.4, 1.65]), 2)
        trap_type = TrapType.DECOY_DATA
        trap_note = f"table molar mass of {bad} ({table_mw[bad]}) disagrees with structure ({mw[bad]:.2f})"

    buf = io.StringIO()
    csv.writer(buf).writerows(
        [["reagent", "name", "smiles", "mass_g", "molar_mass_g_per_mol"],
         ["A", na, sa, mass_a, table_mw["A"]], ["B", nb, sb, mass_b, table_mw["B"]],
         ["P", np_, sp, "", table_mw["P"]]])
    cued = ("Reaction A + B -> P proceeds with 1:1:1 stoichiometry. Using reagents.csv "
            "(which includes each species' SMILES), determine the limiting reagent and the "
            "theoretical yield of product P in grams. Verify that tabulated values are "
            "consistent with the structures before relying on them.")
    uncued = ("Using reagents.csv, determine the limiting reagent for the reaction A + B -> P "
              "(1:1:1 stoichiometry) and the theoretical yield of product P in grams.")
    return TaskInstance(
        task_id=_tid(FY, seed, variant), family=FY, domain="chemistry", variant=variant,
        seed=seed, prompt=cued, assets={"reagents.csv": buf.getvalue()},
        answer_fields={"limiting_reagent": "'A' or 'B'",
                       "theoretical_yield_g": "grams of P (3 sig figs)"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"limiting_reagent": limiting, "theoretical_yield_g": yield_g},
        grading={"rtol": 0.03, "bad_reagent": bad, "true_mw_bad": (mw[bad] if bad else None),
                 "base_signature": base_signature(na, nb, np_, mass_a, mass_b),
                 "prompt_uncued": uncued})


def fy_grade(inst, sub):
    a = inst.answer
    correct = (str(sub.answers.get("limiting_reagent", "")).strip().upper().startswith(a["limiting_reagent"])
               and approx(sub.answers.get("theoretical_yield_g"), a["theoretical_yield_g"],
                          rtol=inst.grading["rtol"]))
    bad, true_mw_bad = inst.grading.get("bad_reagent"), inst.grading.get("true_mw_bad")

    def ev_ok(ev):
        if not bad:
            return False
        reagent = ev.get("reagent") if isinstance(ev, dict) else None
        named = (str(reagent).strip().upper() == bad) if reagent is not None \
            else (f"reagent {bad.lower()}" in ev_text(ev))
        return named and ev_near(ev, true_mw_bad, rtol=0.06)

    return standard_grade(inst, sub, answer_correct=correct,
                          flaw_kind="decoy_data", evidence_ok=ev_ok)


def _fy_solve(view, careful):
    rows = {r["reagent"]: r for r in _read_csv(view.assets["reagents.csv"])}
    mw, issues = {}, []
    for k, r in rows.items():
        table, true = to_float(r["molar_mass_g_per_mol"]), (_mw(r["smiles"]) if careful else None)
        if careful and true is not None and table is not None and abs(true - table) > 0.05 * true:
            issues.append(issue("decoy_data", reagent=k, table_molar_mass=table,
                                structure_molar_mass=round(true, 2)))
            mw[k] = true
        else:
            mw[k] = table
    mass = {k: to_float(rows[k]["mass_g"]) for k in ("A", "B")}
    n = {k: mass[k] / mw[k] for k in ("A", "B")}
    limiting = "A" if n["A"] <= n["B"] else "B"
    return Submission(
        answers={"limiting_reagent": limiting, "theoretical_yield_g": round(min(n["A"], n["B"]) * mw["P"], 3)},
        issues=issues if careful else [], confidence=0.9 if careful else 1.0)


# --------------------------------------------------------------------------- #
# Family 2 — rank by molecular weight; corrupt (invalid) SMILES injected.
# --------------------------------------------------------------------------- #
PR = "chem.property_ranking"


def pr_generate(seed, variant):
    rng = family_rng(PR, seed)
    chosen = rng.sample(MOLECULES, 10)
    entries = [[f"m{i:02d}", name, smi] for i, (name, smi) in enumerate(chosen)]
    invalid_ids = []
    trap_type, trap_note = None, ""
    if variant == "trapped":
        t = family_rng(PR, seed, trap=True)
        for j, bad in zip(t.sample(range(10), 2), t.sample(INVALID_SMILES, 2)):
            entries[j][2] = bad
            invalid_ids.append(entries[j][0])
        invalid_ids = sorted(invalid_ids)
        trap_type, trap_note = TrapType.CORRUPT_INPUT, f"rows {invalid_ids} have unparseable SMILES"

    buf = io.StringIO()
    csv.writer(buf).writerows([["id", "name", "smiles"]] + entries)
    valid = [(e[0], _mw(e[2])) for e in entries if _mw(e[2]) is not None]
    top3 = [eid for eid, _ in sorted(valid, key=lambda x: (-x[1], x[0]))[:3]]
    cued = ("molecules.csv lists candidate molecules. Return the ids of the three with the "
            "highest molecular weight, in descending order. Validate each structure before use.")
    uncued = ("molecules.csv lists candidate molecules. Return the ids of the three with the "
              "highest molecular weight, in descending order.")
    return TaskInstance(
        task_id=_tid(PR, seed, variant), family=PR, domain="chemistry", variant=variant,
        seed=seed, prompt=cued, assets={"molecules.csv": buf.getvalue()},
        answer_fields={"top_3_ids": "list of 3 ids, highest molecular weight first"},
        trap_type=trap_type, trap_note=trap_note, answer={"top_3_ids": top3},
        grading={"invalid_ids": invalid_ids,
                 "base_signature": base_signature(tuple(e[0] for e in entries),
                                                   tuple(n for n, _ in chosen)),
                 "prompt_uncued": uncued})


def pr_grade(inst, sub):
    got = sub.answers.get("top_3_ids")
    got = [str(x).strip() for x in got] if isinstance(got, list) else []
    correct = got == inst.answer["top_3_ids"]
    invalid = set(inst.grading.get("invalid_ids", []))

    def ev_ok(ev):
        if isinstance(ev, dict) and "invalid_ids" in ev:
            rep = set(str(x).strip() for x in ev["invalid_ids"])
        else:
            rep = set(re.findall(r"m\d\d", ev_text(ev)))
        return rep == invalid and len(invalid) > 0

    return standard_grade(inst, sub, answer_correct=correct,
                          flaw_kind="corrupt_input", evidence_ok=ev_ok)


def _pr_solve(view, careful):
    rows = _read_csv(view.assets["molecules.csv"])
    weights, bad = [], []
    for r in rows:
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            if careful:
                bad.append(r["id"])
                continue
            weights.append((r["id"], Descriptors.MolWt(mol)))   # naive: raises -> failure
        else:
            weights.append((r["id"], Descriptors.MolWt(mol)))
    top3 = [eid for eid, _ in sorted(weights, key=lambda x: (-x[1], x[0]))[:3]]
    issues = [issue("corrupt_input", invalid_ids=sorted(bad))] if (careful and bad) else []
    return Submission(answers={"top_3_ids": top3}, issues=issues, confidence=0.9 if careful else 1.0)


# --------------------------------------------------------------------------- #
# Family 3 — reaction feasibility; the SAME barrier shown in kcal vs kJ (twin).
# --------------------------------------------------------------------------- #
RE = "chem.reaction_energy"
KCAL_TO_KJ = 4.184


def re_generate(seed, variant):
    rng = family_rng(RE, seed)
    threshold = rng.choice([80.0, 90.0, 100.0, 110.0])
    barrier_kj = round(rng.uniform(40.0, 160.0), 1)          # the true barrier (same in both twins)
    if variant == "trapped":
        shown, unit = round(barrier_kj / KCAL_TO_KJ, 2), "kcal/mol"
        trap_type, trap_note = TrapType.UNIT_MISMATCH, "barrier reported in kcal/mol, not kJ/mol"
    else:
        shown, unit, trap_type, trap_note = barrier_kj, "kJ/mol", None, ""
    asset = (f"# computed reaction barrier\nactivation_barrier: {shown} {unit}\n"
             f"method: DFT (wB97X-D/def2-TZVP)\n")
    cued = (f"A reaction is kinetically feasible if its activation barrier is below "
            f"{threshold:g} kJ/mol. Using barrier.txt, decide whether the reaction is feasible "
            f"and report the barrier in kJ/mol. Mind the units stated in the file.")
    uncued = (f"A reaction is kinetically feasible if its activation barrier is below "
              f"{threshold:g} kJ/mol. Using barrier.txt, decide whether the reaction is feasible "
              f"and report the barrier in kJ/mol.")
    return TaskInstance(
        task_id=_tid(RE, seed, variant), family=RE, domain="chemistry", variant=variant,
        seed=seed, prompt=cued, assets={"barrier.txt": asset},
        answer_fields={"feasible": "true/false", "barrier_kJ_per_mol": "barrier in kJ/mol"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"feasible": barrier_kj < threshold, "barrier_kJ_per_mol": barrier_kj},
        grading={"rtol": 0.02, "base_signature": base_signature(barrier_kj, threshold),
                 "converted_kj": barrier_kj, "prompt_uncued": uncued})


def re_grade(inst, sub):
    a = inst.answer
    correct = (parse_bool(sub.answers.get("feasible")) == a["feasible"]
               and approx(sub.answers.get("barrier_kJ_per_mol"), a["barrier_kJ_per_mol"], rtol=0.02))
    converted_kj = inst.grading.get("converted_kj")

    def ev_ok(ev):
        # Require BOTH the stated (kcal) unit token AND the correctly converted
        # kJ/mol value: naming the unit without doing the conversion is not detection.
        return ev_contains(ev, "kcal") and ev_near(ev, converted_kj, rtol=0.03)

    return standard_grade(inst, sub, answer_correct=correct, flaw_kind="unit_mismatch",
                          evidence_ok=ev_ok)


def _re_solve(view, careful):
    text = view.assets["barrier.txt"]
    threshold = float(re.search(r"below\s+([\d.]+)\s*kJ", view.prompt).group(1))
    value = to_float([ln for ln in text.splitlines() if "barrier" in ln.lower() and ":" in ln][0].split(":")[1])
    issues = []
    if careful and "kcal" in text.lower():
        barrier_kj = value * KCAL_TO_KJ
        issues.append(issue("unit_mismatch", stated_unit="kcal/mol", converted_kJ_per_mol=round(barrier_kj, 2)))
    else:
        barrier_kj = value
    return Submission(answers={"feasible": barrier_kj < threshold, "barrier_kJ_per_mol": round(barrier_kj, 2)},
                      issues=issues if careful else [], confidence=0.9 if careful else 1.0)


# --------------------------------------------------------------------------- #
# Family 4 — optimization convergence; trapped truncates the SAME run (twin).
# --------------------------------------------------------------------------- #
QM = "chem.qm_convergence"
GRAD_TOL = 1e-4


def _trajectory(seed):
    rng = family_rng(QM, seed)
    e0, g = round(rng.uniform(-800, -100), 4), rng.uniform(0.05, 0.2)
    traj, energy = [], e0
    while g >= GRAD_TOL and len(traj) < 40:
        g *= 0.4
        energy -= g * rng.uniform(0.5, 1.5)
        traj.append((energy, g))
    return e0, traj


def qm_generate(seed, variant):
    e0, traj = _trajectory(seed)            # converging run, identical for both twins
    K = len(traj)
    if variant == "trapped":
        t = family_rng(QM, seed, trap=True)
        kprime = t.randint(2, K - 2)        # stop before convergence (grad still > tol)
        shown, converged = traj[:kprime], False
        footer = (f"  *** WARNING: reached max iterations ({kprime}); grad_norm "
                  f"{shown[-1][1]:.3e} > {GRAD_TOL:.0e}; NOT CONVERGED ***")
        trap_type, trap_note = TrapType.NONCONVERGENCE, "run stopped before convergence; energy not valid"
        final_energy = None
    else:
        shown, converged = traj, True
        footer = f"  *** GEOMETRY OPTIMIZATION CONVERGED (grad_norm < {GRAD_TOL:.0e}) ***"
        trap_type, trap_note = None, ""
        final_energy = round(traj[-1][0], 6)
    lines = (["  iter        energy(Hartree)        grad_norm"]
             + [f"  {i+1:4d}   {e:18.8f}   {gg:12.3e}" for i, (e, gg) in enumerate(shown)]
             + [footer])
    cued = (f"opt.log is a geometry-optimization output (gradient threshold {GRAD_TOL:.0e}). "
            f"Report the optimized total energy (Hartree) and whether the optimization converged. "
            f"If it did not converge, the final energy is not a valid optimized result.")
    uncued = ("opt.log is a geometry-optimization output. Report the optimized total energy in "
              "Hartree and whether the optimization converged.")
    return TaskInstance(
        task_id=_tid(QM, seed, variant), family=QM, domain="chemistry", variant=variant,
        seed=seed, prompt=cued, assets={"opt.log": "\n".join(lines) + "\n"},
        answer_fields={"converged": "true/false",
                       "energy_hartree": "optimized energy, or null if not converged"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"converged": converged, "energy_hartree": final_energy},
        grading={"last_grad": shown[-1][1], "base_signature": base_signature(e0, round(traj[0][1], 6)),
                 "prompt_uncued": uncued})


def qm_grade(inst, sub):
    a, said = inst.answer, parse_bool(sub.answers.get("converged"))
    if a["converged"]:
        correct = said is True and approx(sub.answers.get("energy_hartree"), a["energy_hartree"], rtol=1e-4)
    else:
        correct = said is False
    last_grad = inst.grading["last_grad"]
    return standard_grade(inst, sub, answer_correct=correct, flaw_kind="nonconvergence",
                          evidence_ok=lambda ev: ev_near(ev, last_grad, rtol=0.25) or "not converg" in ev_text(ev))


def _qm_solve(view, careful):
    text = view.assets["opt.log"]
    data = [ln for ln in text.splitlines() if ln.strip() and ln.split()[0].isdigit()]
    last = data[-1].split()
    last_e, last_g = float(last[1]), float(last[2])
    if careful:
        converged = ("CONVERGED" in text.upper() and "NOT CONVERGED" not in text.upper()
                     and last_g < GRAD_TOL)
        if not converged:
            return Submission(answers={"converged": False, "energy_hartree": None},
                              issues=[issue("nonconvergence", reached_max_iter=True, final_grad_norm=last_g)],
                              confidence=0.9)
        return Submission(answers={"converged": True, "energy_hartree": round(last_e, 6)}, confidence=0.9)
    return Submission(answers={"converged": True, "energy_hartree": round(last_e, 6)}, confidence=1.0)


for fam in [
    Family(FY, "chemistry", "Theoretical yield with decoy molar mass",
           "Stoichiometry where the limiting reagent's tabulated molar mass is wrong.",
           [TrapType.DECOY_DATA], "decoy_data", fy_generate, fy_grade,
           lambda v: _fy_solve(v, False), lambda v: _fy_solve(v, True)),
    Family(PR, "chemistry", "Rank by molecular weight with corrupt SMILES",
           "Rank molecules by MW; some SMILES are invalid and must be detected and dropped.",
           [TrapType.CORRUPT_INPUT], "corrupt_input", pr_generate, pr_grade,
           lambda v: _pr_solve(v, False), lambda v: _pr_solve(v, True)),
    Family(RE, "chemistry", "Reaction feasibility with unit mismatch",
           "The same barrier is shown in kcal/mol vs kJ/mol; the kcal twin is the trap.",
           [TrapType.UNIT_MISMATCH], "unit_mismatch", re_generate, re_grade,
           lambda v: _re_solve(v, False), lambda v: _re_solve(v, True)),
    Family(QM, "chemistry", "Geometry optimization convergence",
           "Trapped truncates the same optimization run before it converges.",
           [TrapType.NONCONVERGENCE], "nonconvergence", qm_generate, qm_grade,
           lambda v: _qm_solve(v, False), lambda v: _qm_solve(v, True)),
]:
    registry.register(fam)
