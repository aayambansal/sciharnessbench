# SciHarnessBench — reference scorecard

SciHarnessBench measures **fake science**: does a scientific agent produce a confident,
well-formatted, and *wrong* result when the data hides a flaw a competent scientist would
catch? Every task ships as a clean/trapped pair; the headline metric is the **gap** between
an agent's accuracy on the two, computed over *paired* families (true counterfactual twins).

The two reference agents use **no model and no API key** — they bound the range a real
system under test falls into. `reference-naive` does the right computation but trusts every
input; `reference-careful` validates inputs and reports structured, evidence-bearing flaws.

_12 seeds x 26 families (23 paired, 3 robustness scenarios) = 624 graded tasks per agent. Self-generated and self-graded._

## Headline (paired families)

| agent | competence | robustness | fake-science gap | confident-wrong | false-alarm | trap detection |
|---|---:|---:|---:|---:|---:|---:|
| `reference-naive` | 100.0% | 0.0% | **100.0 pts** | 92.4% | 0.0% | 0.0% |
| `reference-careful` | 100.0% | 100.0% | **0.0 pts** | 0.0% | 0.0% | 100.0% |

Naive fake-science gap 95% bootstrap CI: [100.0, 100.0] pts over 276 pairs. Both agents are equally competent; they diverge entirely on
robustness. The gap is a within-task difference (robust to grading noise) and cannot be gamed
by abstaining (a flag/abstention on a clean task is a false alarm that scores zero).

## Robustness by domain (paired)

| domain | competence | naive robustness | careful robustness |
|---|---:|---:|---:|
| astronomy | 100.0 | 0.0 | 100.0 |
| biology | 100.0 | 0.0 | 100.0 |
| chemistry | 100.0 | 0.0 | 100.0 |
| climate | 100.0 | 0.0 | 100.0 |
| materials | 100.0 | 0.0 | 100.0 |
| neuroscience | 100.0 | 0.0 | 100.0 |
| physics | 100.0 | 0.0 | 100.0 |
| statistics | 100.0 | 0.0 | 100.0 |

## Robustness by trap type (paired)

| trap type | naive robustness | careful robustness | careful trap detection | n |
|---|---:|---:|---:|---:|
| confounding | 0.0 | 100.0 | 100.0 | 24 |
| corrupt_input | 0.0 | 100.0 | 100.0 | 48 |
| data_leakage | 0.0 | 100.0 | 100.0 | 12 |
| decoy_data | 0.0 | 100.0 | 100.0 | 24 |
| extrapolation | 0.0 | 100.0 | 100.0 | 24 |
| future_leakage | 0.0 | 100.0 | 100.0 | 12 |
| nonconvergence | 0.0 | 100.0 | 100.0 | 36 |
| underpowered | 0.0 | 100.0 | 100.0 | 24 |
| unit_mismatch | 0.0 | 100.0 | 100.0 | 60 |
| wrong_control | 0.0 | 100.0 | 100.0 | 12 |

## Robustness scenarios (non-paired families)

| family | naive robustness | careful robustness |
|---|---:|---:|
| neuro.channel_responsiveness | 0.0 | 100.0 |
| neuro.decoding_circular | 0.0 | 100.0 |
| stats.multiple_comparisons | 0.0 | 100.0 |

## Task families (26)


**astronomy**
- `astro.aperture_photometry` — Aperture photometry with corrupt pixels _(trap: corrupt_input)_
- `astro.brightness_threshold` — Brightness threshold with unit mismatch _(trap: unit_mismatch)_
- `astro.transit_fit` — Transit-depth fit (convergence) _(trap: nonconvergence)_

**biology**
- `bio.diffexpr_confounding` — Differential expression with batch confounding _(trap: confounding)_
- `bio.gc_content` — Mean GC content with corrupt FASTA records _(trap: corrupt_input)_
- `bio.wrong_control` — Treatment vs the tissue-matched control _(trap: wrong_control)_

**chemistry**
- `chem.property_ranking` — Rank by molecular weight with corrupt SMILES _(trap: corrupt_input)_
- `chem.qm_convergence` — Geometry optimization convergence _(trap: nonconvergence)_
- `chem.reaction_energy` — Reaction feasibility with unit mismatch _(trap: unit_mismatch)_
- `chem.theoretical_yield` — Theoretical yield with decoy molar mass _(trap: decoy_data)_

**climate**
- `climate.forecast_skill` — Forecast skill with future leakage _(trap: future_leakage)_
- `climate.regional_threshold` — Regional mean vs degC threshold with unit mismatch _(trap: unit_mismatch)_
- `climate.station_mean` — Regional mean over stations with a decoy station _(trap: decoy_data)_

**materials**
- `mat.alloy_bandgap` — Alloy bandgap extrapolation _(trap: extrapolation)_
- `mat.cell_density` — Densest crystal with corrupt structures _(trap: corrupt_input)_
- `mat.formation_stability` — Formation-energy stability with unit mismatch _(trap: unit_mismatch)_

**neuroscience**
- `neuro.channel_responsiveness` — Channel responsiveness with multiple comparisons _(trap: multiple_comparisons)_ _(scenario)_
- `neuro.decoding_circular` — Neural decoding without double dipping _(trap: circular_analysis)_ _(scenario)_
- `neuro.tuning_difference` — Firing-rate difference from an adequate sample _(trap: underpowered)_

**physics**
- `phys.heat_cfl` — Heat-equation CFL stability _(trap: nonconvergence)_
- `phys.kinetic_threshold` — Kinetic-energy threshold with unit mismatch _(trap: unit_mismatch)_
- `phys.smallangle_extrapolation` — Small-angle fit extrapolation _(trap: extrapolation)_

**statistics**
- `stats.data_leakage` — Honest AUC with target leakage _(trap: data_leakage)_
- `stats.multiple_comparisons` — Multiple comparisons _(trap: multiple_comparisons)_ _(scenario)_
- `stats.simpson` — Simpson's paradox _(trap: confounding)_
- `stats.underpowered` — Underpowered conclusion _(trap: underpowered)_
