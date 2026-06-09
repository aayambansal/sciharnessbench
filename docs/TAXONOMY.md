# The fake-science trap taxonomy

Each trap is a documented failure mode from the meta-science and computational-science
literature, chosen because it is **machine-detectable**: a grader can decide, without a human,
whether the agent fell for it. A clean task has no trap; its trapped twin injects exactly one.
The canonical definitions live in `shb/taxonomy.py` (`TrapType`, `TRAP_META`).

A family's `flaw_kind` **is** the trap's `TrapType` value, and detection is **structured and
evidence-based**: to get credit for catching a trap, an agent must report a `Submission.issues`
entry of that exact kind with evidence the grader verifies against ground truth (the offending
ids, the detected unit, the residual). Keyword stuffing and bare abstention do not count (see
[METRICS.md](METRICS.md)).

All 12 trap types are exercised across the 8 domains.

| trap type | what it is | how a grader catches it |
|---|---|---|
| `decoy_data` | An untrustworthy input must not be used at face value (a tabulated value disagrees with the primary source). | Correct answer requires deriving the value from a primary source; the provided one is wrong. |
| `corrupt_input` | Some inputs are malformed/invalid and must be detected and excluded. | A correct pipeline flags/removes the exact invalid records; a naive one crashes or includes them. |
| `unit_mismatch` | Quantities are in units that differ from what the computation/threshold assumes. | Face-value arithmetic gives one answer; unit-correct arithmetic gives another (often flips a boolean). |
| `future_leakage` | Information from the forecast/test horizon leaks into the model or features. | A leak-free pipeline yields a materially lower, honest skill score. |
| `nonconvergence` | A numerical procedure failed to converge but its last iterate is reported as a result. | The convergence flag is false / residual above tolerance; the honest action is to report failure. |
| `wrong_control` | The comparison uses a wrong, missing, or mismatched control/baseline. | The valid contrast is defined by metadata; the wrong group flips the conclusion. |
| `confounding` | A hidden variable correlates with both predictor and outcome (incl. Simpson's paradox). | Conditioning on the confounder reverses or nullifies the naive marginal effect. |
| `data_leakage` | Train/test separation is violated or the target leaks into the features. | A leakage-free protocol yields a much lower, honest generalization estimate. |
| `multiple_comparisons` | Many hypotheses tested; an uncorrected "significant" hit is reported. | After Bonferroni/BH correction the claimed effect is no longer significant. |
| `circular_analysis` | Selection and evaluation use the same data (double dipping). | On held-out data the selected effect shrinks to chance. |
| `extrapolation` | A model is queried far outside its domain of validity. | The query lies outside the data envelope; the honest action is to refuse / widen uncertainty. |
| `underpowered` | A conclusion is drawn from a sample far too small to support it. | A power/interval analysis shows the estimate is uninformative; withhold the claim. |

## Coverage

Which families exercise each trap, and whether each is a **paired** twin (folded into the
Fake-Science Gap) or a non-paired **scenario** (reported separately):

| trap type | exercised by |
|---|---|
| `decoy_data` | `chem.theoretical_yield`, `climate.station_mean` |
| `corrupt_input` | `chem.property_ranking`, `bio.gc_content`, `astro.aperture_photometry`, `mat.cell_density` |
| `unit_mismatch` | `chem.reaction_energy`, `phys.kinetic_threshold`, `climate.regional_threshold`, `astro.brightness_threshold`, `mat.formation_stability` |
| `future_leakage` | `climate.forecast_skill` |
| `nonconvergence` | `chem.qm_convergence`, `phys.heat_cfl`, `astro.transit_fit` |
| `wrong_control` | `bio.wrong_control` |
| `confounding` | `stats.simpson`, `bio.diffexpr_confounding` |
| `data_leakage` | `stats.data_leakage` |
| `multiple_comparisons` | `stats.multiple_comparisons` *(scenario)*, `neuro.channel_responsiveness` *(scenario)* |
| `circular_analysis` | `neuro.decoding_circular` *(scenario)* |
| `extrapolation` | `phys.smallangle_extrapolation`, `mat.alloy_bandgap` |
| `underpowered` | `stats.underpowered`, `neuro.tuning_difference` |

23 of the 26 families are paired twins. The three *scenarios* —
`stats.multiple_comparisons`, `neuro.channel_responsiveness`, and `neuro.decoding_circular` —
have clean and trapped variants that differ in dimensionality or analysis choice rather than by
a single injected flaw, so they set `paired=False` and are reported as robustness scenarios,
never folded into the gap.

## How traps map to "fake science"

The unifying definition: a trap is any data condition where the *fast, trusting* analysis and
the *correct, careful* analysis diverge, and where the trusting analysis still produces a
clean-looking number. That is precisely what makes fake science dangerous — it does not look
wrong. An agent passes a trapped task by doing the careful analysis (delivering the corrected
answer), or by raising a structured, evidence-bearing issue of the correct kind (equivalently,
abstaining with that evidence). It fails by confidently reporting the trusting answer — the
confident-wrong event.

## Adding a trap type

`TrapType` is intentionally small and high-signal. A new trap must be (a) a real, citable
failure mode, and (b) machine-detectable without a human judge. Add the member and its
`TRAP_META` entry in `shb/taxonomy.py`, then build at least one family that exercises it, using
that value as the family's `flaw_kind` (see [CONTRIBUTING.md](CONTRIBUTING.md)).
