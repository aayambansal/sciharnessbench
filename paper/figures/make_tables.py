#!/usr/bin/env python3
"""Emit the paper's LaTeX tables/macros from the live registry + v2 scorecard JSON.

Keeps every count and result in the paper in sync with the benchmark. Writes
.tex fragments into ../tables/ that the paper \\input{}s.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TABLES = os.path.join(os.path.dirname(HERE), "tables")
os.makedirs(TABLES, exist_ok=True)
sys.path.insert(0, REPO)

from shb import registry  # noqa: E402

SCORE = os.path.join(REPO, "results")
naive = json.load(open(os.path.join(SCORE, "scorecard_reference-naive.json")))
careful = json.load(open(os.path.join(SCORE, "scorecard_reference-careful.json")))

TRAP_SHORT = {
    "decoy_data": "decoy data", "corrupt_input": "corrupt input",
    "unit_mismatch": "unit mismatch", "future_leakage": "future leakage",
    "nonconvergence": "non-convergence", "wrong_control": "wrong control",
    "confounding": "confounding", "data_leakage": "data leakage",
    "multiple_comparisons": "multiple comparisons", "circular_analysis": "circular analysis",
    "extrapolation": "extrapolation", "underpowered": "underpowered",
}


def w(name, body):
    open(os.path.join(TABLES, name), "w").write(body)
    print("wrote tables/" + name)


def composition():
    fams = registry.all_families()
    rows = []
    for dom in sorted({f.domain for f in fams}):
        dfams = [f for f in fams if f.domain == dom]
        traps = sorted({t.value for f in dfams for t in f.trap_types})
        rows.append(f"{dom} & {len(dfams)} & {', '.join(TRAP_SHORT[t] for t in traps)} \\\\")
    ntr = len({t.value for f in fams for t in f.trap_types})
    body = ("\\begin{tabular}{@{}lcl@{}}\n\\toprule\n"
            "Domain & Families & Trap types exercised \\\\\n\\midrule\n" + "\n".join(rows)
            + f"\n\\midrule\n\\textbf{{Total}} & \\textbf{{{len(fams)}}} & "
              f"\\textbf{{{ntr} distinct trap types}} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
    w("tab_composition.tex", body)


def reference():
    def row(label, c):
        h = c["headline"]
        return (f"{label} & {100*h['competence']:.1f} & {100*h['robustness']:.1f} & "
                f"{100*h['fake_science_gap']:.1f} & {100*h['confident_wrong_rate']:.1f} & "
                f"{100*h['false_alarm_rate']:.1f} & {100*h['trap_detection_rate']:.1f} \\\\")
    body = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
            "Reference agent & Comp. & Robust. & Gap & Conf.-wrong & False-alarm & Trap det. \\\\\n"
            "\\midrule\n" + row("naive (trusts inputs)", naive) + "\n"
            + row("careful (validates)", careful) + "\n\\bottomrule\n\\end{tabular}\n")
    w("tab_reference.tex", body)


def by_domain():
    rows = []
    for d in sorted(careful["by_domain"]):
        bn, bc = naive["by_domain"][d], careful["by_domain"][d]
        rows.append(f"{d} & {100*bc['competence']:.0f} & {100*bn['robustness']:.0f} & "
                    f"{100*bc['robustness']:.0f} & {bc['n_trapped']} \\\\")
    body = ("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
            "Domain & Comp. & Naive robust. & Careful robust. & Trapped tasks \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w("tab_bydomain.tex", body)


DOMAIN_LIB = {
    "chemistry": "RDKit", "statistics": "scipy/sklearn", "physics": "numpy/scipy",
    "biology": "Biopython/scipy", "climate": "numpy", "astronomy": "astropy/scipy",
    "neuroscience": "numpy/scipy", "materials": "numpy",
}
TRAP_MOTIV = {
    "decoy_data": "ioannidis2005why", "corrupt_input": "leek2010batch",
    "unit_mismatch": "tropsha2010qsar", "future_leakage": "kapoor2023leakage",
    "nonconvergence": "courant1928cfl", "wrong_control": "ioannidis2005why",
    "confounding": "blyth1972simpson", "data_leakage": "kapoor2023leakage",
    "multiple_comparisons": "benjamini1995fdr", "circular_analysis": "kriegeskorte2009circular",
    "extrapolation": "tropsha2010qsar", "underpowered": "button2013power",
}


def families_spec():
    fams = registry.all_families()
    esc = "\\_"
    rows = []
    for f in sorted(fams, key=lambda x: x.family_id):
        fid = f.family_id.replace("_", esc)
        paired = "yes" if f.paired else "scenario"
        rows.append(f"\\texttt{{{fid}}} & {DOMAIN_LIB.get(f.domain, '-')} & "
                    f"{TRAP_SHORT[f.flaw_kind]} & {paired} & {f.title} \\\\")
    body = ("\\begin{tabular}{@{}lllll@{}}\n\\toprule\n"
            "Family & Library & Trap & Paired & Task \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w("tab_families.tex", body)


def taxonomy():
    fams = registry.all_families()
    rows = []
    for trap in sorted(TRAP_SHORT):
        nfam = sum(1 for f in fams for t in f.trap_types if t.value == trap)
        rows.append(f"{TRAP_SHORT[trap]} & \\citep{{{TRAP_MOTIV[trap]}}} & {nfam} \\\\")
    body = ("\\begin{tabular}{@{}llc@{}}\n\\toprule\n"
            "Trap type & Literature motivation & Families \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w("tab_taxonomy.tex", body)


def facts():
    fams = registry.all_families()
    npaired = sum(1 for f in fams if f.paired)
    h_n, h_c = naive["headline"], careful["headline"]
    ci = h_n.get("gap_ci") or {}
    defs = {
        "NumDomains": len({f.domain for f in fams}), "NumFamilies": len(fams),
        "NumPaired": npaired, "NumScenario": len(fams) - npaired,
        "NumTrapTypes": len({t.value for f in fams for t in f.trap_types}),
        "SeedsPerFamily": h_c["n_trapped"] // max(npaired, 1),
        "TotalTasksPerAgent": careful["n_grades"],
        "NaiveCompetence": f"{100*h_n['competence']:.1f}", "NaiveRobustness": f"{100*h_n['robustness']:.1f}",
        "NaiveGap": f"{100*h_n['fake_science_gap']:.1f}", "NaiveConfidentWrong": f"{100*h_n['confident_wrong_rate']:.1f}",
        "CarefulCompetence": f"{100*h_c['competence']:.1f}", "CarefulRobustness": f"{100*h_c['robustness']:.1f}",
        "CarefulGap": f"{100*h_c['fake_science_gap']:.1f}", "CarefulTrapDetection": f"{100*h_c['trap_detection_rate']:.1f}",
        "CarefulFalseAlarm": f"{100*h_c['false_alarm_rate']:.1f}",
        "NaiveGapCIlow": f"{100*ci.get('ci95_low', 0):.1f}", "NaiveGapCIhigh": f"{100*ci.get('ci95_high', 0):.1f}",
    }
    body = "% AUTO-GENERATED by make_tables.py — do not edit by hand.\n"
    body += "".join(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n" for k, v in defs.items())
    w("facts.tex", body)


if __name__ == "__main__":
    composition()
    reference()
    by_domain()
    families_spec()
    taxonomy()
    facts()
