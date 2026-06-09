# SciHarnessBench

**The fake-science benchmark for AI science agents.**

Most AI-for-science benchmarks ask *can a model answer a science question* or *can it complete
a workflow*. SciHarnessBench asks the question that predicts whether you can trust an
autonomous scientific agent:

> When the data hides a flaw a competent scientist would catch, does the agent catch it —
> or does it produce a confident, well-formatted, and **wrong** result?

Every task ships as a **clean/trapped pair**: the same problem, once clean and once with a
single planted flaw (a decoy value, a unit mismatch, a leaked label, a non-converged run, a
confound, an uncorrected multiple comparison). For *paired* families the two versions are
genuine counterfactual twins — identical except the flaw — so the headline metric, the
**Fake-Science Gap** (clean minus trapped accuracy), isolates the trap.

Tasks are **procedurally instantiated from human-authored templates and graded by
deterministic code** — no human annotates instances at evaluation time, and there is no LLM
judge. Ground truth is computed by real scientific libraries (RDKit, scipy, scikit-learn,
Biopython, astropy). This makes the suite cheap to scale across domains and
**contamination-resistant**: instances are regenerable under a private salt.

## Headline result

Two built-in reference agents (no model, no API key) bound the achievable range. Both run the
same underlying computation; they differ only in whether they validate their inputs. Metrics
are over the 23 paired families (12 seeds each); detection requires a structured,
evidence-bearing flaw report.

| agent | competence | robustness | fake-science gap | confident-wrong | false-alarm | trap detection |
|---|---:|---:|---:|---:|---:|---:|
| `reference-naive` (trusts inputs) | **100%** | **0%** | **100 pts** [CI 100–100] | 92% | 0% | 0% |
| `reference-careful` (validates) | **100%** | **100%** | **0 pts** | 0% | 0% | 100% |

Both agents are equally competent on clean tasks, so the science is doable. The naive agent
then falls for essentially every trap — 92% as *confident wrong* answers, the rest as crashes
on corrupt input (an unsafe failure we score separately, not as fake science). The careful
agent shrugs the traps off without ever false-alarming on a clean task. A real system under
test lands between these lines, and where it lands is the finding.

## Why the metric is honest and hard to game

- **Within-task difference.** The gap is clean-minus-trapped on the *same* problems, robust to
  grading noise, with a paired bootstrap 95% CI; there is no LLM judge to drift.
- **Abstention-proof.** A flaw claim or abstention on a clean task is a *false alarm* scoring
  zero, so "always flag" / "always abstain" destroys competence. You must be competent **and**
  robust.
- **Evidence-gated detection.** Passing a trap by detection requires a *structured* issue of
  the correct kind with evidence the grader checks against ground truth — keyword stuffing
  fails (`tests/test_adversarial.py`).
- **No variant leak.** The agent sees an opaque task id and opaque file paths; the
  clean/trapped label and seed are never visible. Twins are never shown adjacently.
- **Contamination-resistant.** A private `SHB_SALT` regenerates instances for the official
  split; memorizing the public dev set buys nothing.

See [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for the full threat model and the isolated,
salted official evaluation protocol.

## Quickstart

```bash
pip install -e .            # numpy, scipy, scikit-learn, pandas, rdkit, biopython, astropy
pytest -q                   # contract, scoring, harness, discrimination, and adversarial tests

# the reference agents (no API key) — see the discrimination
python scripts/run_benchmark.py --agent both --seeds 12
python scripts/run_benchmark.py --agent both --seeds 12 --domains chemistry physics

# regenerate the reference scorecard under results/
python scripts/make_scorecard.py
```

### Testing your own model

Wrap any model or agent loop in a `complete(prompt) -> str` callable and hand it to
`LLMAgent`, which renders each task and asks for a strict-JSON answer with a structured
`issues` channel (`{"kind", "evidence"}`) and a `confidence`:

```python
from shb import run_benchmark, aggregate
from shb.agents import LLMAgent

agent = LLMAgent(complete=my_model, name="my-model")
print(aggregate("my-model", run_benchmark(agent, seeds=range(12))).to_markdown())
```

For an official, leaderboard-grade score, run under the isolated, salted protocol in
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) and [docs/LEADERBOARD.md](docs/LEADERBOARD.md).

## Coverage

**8 domains · 26 task families · all 12 trap types** (23 paired families + 3 non-paired
robustness scenarios). Each family is grounded in a real library.

| domain | families | example traps |
|---|---|---|
| chemistry | 4 | decoy molar mass, corrupt SMILES, kcal/kJ unit mismatch, QM non-convergence |
| statistics | 4 | target leakage, Simpson's paradox, multiple comparisons*, underpowered n |
| physics | 3 | CFL instability, unit mismatch, out-of-domain extrapolation |
| biology | 3 | batch confounding, wrong control, corrupt FASTA |
| climate | 3 | future leakage, Kelvin/Celsius mismatch, decoy station |
| astronomy | 3 | corrupt pixels, non-converged fit, magnitude/flux mismatch |
| neuroscience | 3 | circular analysis*, multiple comparisons*, underpowered |
| materials | 3 | non-physical crystal, eV/meV mismatch, composition extrapolation |

`*` non-paired robustness scenarios (clean and trapped differ in dimensionality or analysis
choice, not by a single injected flaw — reported separately, never folded into the gap). Full
taxonomy: [docs/TAXONOMY.md](docs/TAXONOMY.md).

## Repository

```
shb/
  taxonomy.py   # the 12 fake-science TrapTypes, grounded in the meta-science literature
  types.py      # contract: opaque AgentView, structured Submission, Grade, Family (paired flag)
  utils.py      # salted RNGs, evidence helpers, standard_grade (the clean/trapped policy)
  runner.py     # opaque workdirs, shuffled order, cued/uncued prompts
  scoring.py    # competence, robustness, Fake-Science Gap + bootstrap CI, confident-wrong, macro/micro
  agents/       # ReferenceAgent (naive/careful oracles) + LLMAgent adapter
  domains/      # one module per domain; each registers its families
scripts/        # run_benchmark.py, make_scorecard.py
paper/          # the AAAI submission (LaTeX), with data-driven figures/tables/facts
tests/          # contract, scoring, harness, discrimination, adversarial
docs/           # SPEC, TAXONOMY, METRICS, THREAT_MODEL, CONTRIBUTING, LEADERBOARD
results/        # the reference scorecard artifact
```

Adding a domain or family is ~100 lines against a fixed contract — see
[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md). Design and grading details:
[docs/SPEC.md](docs/SPEC.md), [docs/METRICS.md](docs/METRICS.md).

## Honest limitations

- **Procedurally generated, grounded in real computation** — not lifted verbatim from specific
  papers. The *methods* and *failure modes* come from the literature (the reproducibility
  crisis, known numerical pitfalls); the instances are synthesized so they can be regenerated
  and graded without a human. A literature-curated track is future work.
- **The reference agents are oracles, not contestants.** `reference-careful` encodes the
  correct method per family; it bounds the discriminative range and documents correct
  behavior. The systems under test are real agents without these solvers.
- **One trap per trapped instance.** Real analyses fail in combinations; v1 isolates single
  failure modes so the gap attributes to a named trap.
- **Single-step tasks.** v1 tasks are focused analyses, not long multi-tool workflows; a
  workflow track is on the roadmap.
- **Model evaluation is forthcoming.** This release validates the benchmark with reference
  oracles; large-scale evaluation of frontier agents will populate the public leaderboard.

## License

MIT — see [LICENSE](LICENSE).
