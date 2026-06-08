# SciHarnessBench — reference scorecard

SciHarnessBench measures **fake science**: does a scientific agent produce a confident,
well-formatted, and *wrong* result when the data hides a flaw a competent scientist would
catch? Every task ships as a clean/trapped pair; the headline metric is the **gap** between
an agent's accuracy on the two.

The two reference agents below use **no model and no API key** — they bound the range a real
system under test falls into:

- `reference-naive` does the right computation but trusts every input at face value → commits the fake science.
- `reference-careful` validates inputs, checks units/convergence/confounds, and corrects or flags the flaw → does real science.

_Generated over 10 seeds × 26 families = 520 graded tasks (clean + trapped), fully self-generated and self-graded._

## Headline

| agent | competence (clean) | robustness (trapped) | fake-science gap | trap detection |
|---|---:|---:|---:|---:|
| `reference-naive` | 100.0% | 0.0% | **100.0 pts** | 0.0% |
| `reference-careful` | 100.0% | 100.0% | **0.0 pts** | 100.0% |

Both agents are equally **competent** (they solve clean tasks). They diverge entirely on
**robustness**: the naive agent collapses under planted flaws while the careful agent holds.
The gap is the product — and because it is a within-task difference, it is robust to grading
noise and cannot be gamed by reflexive abstention (abstaining on a clean task scores zero).

## Robustness by domain (trapped pass rate)

| domain | naive | careful |
|---|---:|---:|
| astronomy | 0.0% | 100.0% |
| biology | 0.0% | 100.0% |
| chemistry | 0.0% | 100.0% |
| climate | 0.0% | 100.0% |
| materials | 0.0% | 100.0% |
| neuroscience | 0.0% | 100.0% |
| physics | 0.0% | 100.0% |
| statistics | 0.0% | 100.0% |

## Robustness by trap type (trapped pass rate)

| trap type | naive | careful | n |
|---|---:|---:|---:|
| circular_analysis | 0.0% | 100.0% | 10 |
| confounding | 0.0% | 100.0% | 20 |
| corrupt_input | 0.0% | 100.0% | 40 |
| data_leakage | 0.0% | 100.0% | 10 |
| decoy_data | 0.0% | 100.0% | 20 |
| extrapolation | 0.0% | 100.0% | 20 |
| future_leakage | 0.0% | 100.0% | 10 |
| multiple_comparisons | 0.0% | 100.0% | 20 |
| nonconvergence | 0.0% | 100.0% | 30 |
| underpowered | 0.0% | 100.0% | 20 |
| unit_mismatch | 0.0% | 100.0% | 50 |
| wrong_control | 0.0% | 100.0% | 10 |

## Task families (26)


**astronomy**
- `astro.aperture_photometry` — Aperture photometry with corrupt pixels _(traps: corrupt_input)_
- `astro.brightness_threshold` — Brightness threshold with unit mismatch _(traps: unit_mismatch)_
- `astro.transit_fit` — Transit-depth fit (convergence) _(traps: nonconvergence)_

**biology**
- `bio.diffexpr_confounding` — Differential expression with batch confounding _(traps: confounding)_
- `bio.gc_content` — Mean GC content with corrupt FASTA records _(traps: corrupt_input)_
- `bio.wrong_control` — Treatment vs the tissue-matched control _(traps: wrong_control)_

**chemistry**
- `chem.property_ranking` — Rank by molecular weight with corrupt SMILES _(traps: corrupt_input)_
- `chem.qm_convergence` — Geometry optimization convergence _(traps: nonconvergence)_
- `chem.reaction_energy` — Reaction feasibility with unit mismatch _(traps: unit_mismatch)_
- `chem.theoretical_yield` — Theoretical yield with decoy molar mass _(traps: decoy_data)_

**climate**
- `climate.forecast_skill` — Forecast skill with future leakage _(traps: future_leakage)_
- `climate.regional_threshold` — Regional mean vs degC threshold with unit mismatch _(traps: unit_mismatch)_
- `climate.station_mean` — Regional mean over stations with a decoy station _(traps: decoy_data)_

**materials**
- `mat.alloy_bandgap` — Alloy bandgap extrapolation _(traps: extrapolation)_
- `mat.cell_density` — Densest crystal with corrupt structures _(traps: corrupt_input)_
- `mat.formation_stability` — Formation-energy stability with unit mismatch _(traps: unit_mismatch)_

**neuroscience**
- `neuro.channel_responsiveness` — Channel responsiveness with multiple comparisons _(traps: multiple_comparisons)_
- `neuro.decoding_circular` — Neural decoding without double dipping _(traps: circular_analysis)_
- `neuro.tuning_difference` — Firing-rate difference from an adequate sample _(traps: underpowered)_

**physics**
- `phys.heat_cfl` — Heat-equation CFL stability _(traps: nonconvergence)_
- `phys.kinetic_threshold` — Kinetic-energy threshold with unit mismatch _(traps: unit_mismatch)_
- `phys.smallangle_extrapolation` — Small-angle fit extrapolation _(traps: extrapolation)_

**statistics**
- `stats.data_leakage` — Honest AUC with target leakage _(traps: data_leakage)_
- `stats.multiple_comparisons` — Multiple comparisons _(traps: multiple_comparisons)_
- `stats.simpson` — Simpson's paradox _(traps: confounding)_
- `stats.underpowered` — Underpowered conclusion _(traps: underpowered)_
