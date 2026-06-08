# The fake-science trap taxonomy

Each trap is a documented failure mode from the meta-science and computational-science
literature, chosen because it is **machine-detectable**: a grader can decide, without a human,
whether the agent fell for it. A clean task has no trap; its trapped twin injects exactly one.
The canonical definitions live in `shb/taxonomy.py` (`TrapType`, `TRAP_META`).

All 12 trap types are exercised by the v1 suite across 8 domains.

| trap type | what it is | how a grader catches it | exercised by |
|---|---|---|---|
| `decoy_data` | An untrustworthy input must not be used at face value (a tabulated value disagrees with the primary source). | Correct answer requires deriving the value from a primary source; the provided one is wrong. | `chem.theoretical_yield`, `climate.station_mean` |
| `corrupt_input` | Some inputs are malformed/invalid and must be detected and excluded. | A correct pipeline flags/removes the exact invalid records; a naive one crashes or includes them. | `chem.property_ranking`, `bio.gc_content`, `astro.aperture_photometry`, `mat.cell_density` |
| `unit_mismatch` | Quantities are in units that differ from what the computation/threshold assumes. | Face-value arithmetic gives one answer; unit-correct arithmetic gives another (often flips a boolean). | `chem.reaction_energy`, `phys.kinetic_threshold`, `climate.regional_threshold`, `astro.brightness_threshold`, `mat.formation_stability` |
| `future_leakage` | Information from the forecast/test horizon leaks into the model or features. | A leak-free pipeline yields a materially lower, honest skill score. | `climate.forecast_skill` |
| `nonconvergence` | A numerical procedure failed to converge but its last iterate is reported as a result. | The convergence flag is false / residual above tolerance; the honest action is to report failure. | `chem.qm_convergence`, `phys.heat_cfl`, `astro.transit_fit` |
| `wrong_control` | The comparison uses a wrong, missing, or mismatched control/baseline. | The valid contrast is defined by metadata; the wrong group flips the conclusion. | `bio.wrong_control` |
| `confounding` | A hidden variable correlates with both predictor and outcome (incl. Simpson's paradox). | Conditioning on the confounder reverses or nullifies the naive marginal effect. | `stats.simpson`, `bio.diffexpr_confounding` |
| `data_leakage` | Train/test separation is violated or the target leaks into the features. | A leakage-free protocol yields a much lower, honest generalization estimate. | `stats.data_leakage` |
| `multiple_comparisons` | Many hypotheses tested; an uncorrected "significant" hit is reported. | After Bonferroni/BH correction the claimed effect is no longer significant. | `stats.multiple_comparisons`, `neuro.channel_responsiveness` |
| `circular_analysis` | Selection and evaluation use the same data (double dipping). | On held-out data the selected effect shrinks to chance. | `neuro.decoding_circular` |
| `extrapolation` | A model is queried far outside its domain of validity. | The query lies outside the data envelope; the honest action is to refuse / widen uncertainty. | `phys.smallangle_extrapolation`, `mat.alloy_bandgap` |
| `underpowered` | A conclusion is drawn from a sample far too small to support it. | A power/interval analysis shows the estimate is uninformative; withhold the claim. | `stats.underpowered`, `neuro.tuning_difference` |

## How traps map to "fake science"

The unifying definition: a trap is any data condition where the *fast, trusting* analysis and
the *correct, careful* analysis diverge, and where the trusting analysis still produces a
clean-looking number. That is precisely what makes fake science dangerous — it does not look
wrong. An agent passes a trapped task by doing the careful analysis (delivering the corrected
answer), or by recognizing the task is not answerable as posed (abstaining / flagging). It
fails by confidently reporting the trusting answer.

## Adding a trap type

`TrapType` is intentionally small and high-signal. A new trap must be (a) a real, citable
failure mode, and (b) machine-detectable without a human judge. Add the member and its
`TRAP_META` entry in `shb/taxonomy.py`, then build at least one family that exercises it (see
[CONTRIBUTING.md](CONTRIBUTING.md)).
