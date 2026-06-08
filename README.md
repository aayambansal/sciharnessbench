# SciHarnessBench

**The fake-science benchmark for AI science agents.**

Existing AI-for-science benchmarks ask *can a model answer a science question* or *can it
run a workflow end to end*. SciHarnessBench asks the question that actually predicts whether
you can trust an autonomous scientific agent:

> When the data hides a flaw a competent scientist would catch, does the agent catch it —
> or does it produce a confident, well-formatted, and **wrong** result?

Every task ships as a **clean/trapped pair**: the same problem, once clean and once with a
single planted flaw (a decoy value, a unit mismatch, a leaked label, a non-converged run, an
uncorrected multiple comparison, a confound). The headline metric is the **gap** between an
agent's accuracy on the clean and trapped versions.

The whole benchmark is **self-generating and self-grading** — no human authors tasks and no
human (and no LLM judge) grades them. Ground truth is computed by real scientific libraries
(RDKit, scipy, scikit-learn, Biopython, astropy); flaws are injected programmatically and
checked by objective code. That makes it cheap to scale across every domain, and
**contamination-resistant**: trapped instances are regenerable, so a memorized public set can
be refreshed with new seeds.

## Headline result

Two built-in reference agents (no model, no API key) bound the achievable range. Both do the
same underlying computation; they differ only in whether they validate their inputs.

| agent | competence (clean) | robustness (trapped) | fake-science gap | trap detection |
|---|---:|---:|---:|---:|
| `reference-naive` (trusts inputs) | **100.0%** | **0.0%** | **100.0 pts** | 0.0% |
| `reference-careful` (validates) | **100.0%** | **100.0%** | **0.0 pts** | 100.0% |

_Over 50 seeds × 26 families = 2,600 graded tasks per agent: the careful agent passes every
clean and every trapped task; the naive agent solves every clean task and falls for **every**
trap._ A real system under test lands between these two lines, and where it lands is the
finding.

## Why the gap metric is honest

- **It is a within-task difference**, not an absolute judge score, so it is robust to grading
  noise and does not drift as models improve.
- **It cannot be gamed by abstaining.** Crying "trap!" on a clean task is a false alarm that
  scores zero, so reflexive abstention tanks competence. The target agent must be *both*
  competent (high clean) *and* robust (high trapped) — exactly the profile of real science.
- **A crash is a failure.** An agent that throws on a corrupt input scores zero on that task;
  refusing to validate is not an excuse.

## Quickstart

```bash
pip install -e .            # numpy, scipy, scikit-learn, pandas, rdkit, biopython, astropy
pytest -q                   # 118 tests, incl. the discrimination invariant

# run the reference agents and see the discrimination (no API key needed)
python scripts/run_benchmark.py --agent both --seeds 5
python scripts/run_benchmark.py --agent both --seeds 5 --domains chemistry physics

# regenerate the full reference scorecard under results/
python scripts/make_scorecard.py
```

### Testing your own model

Wrap any model — an API call, a local model, or a full tool-using agent loop — in a
`complete(prompt) -> str` callable and hand it to `LLMAgent`:

```python
from shb import run_benchmark, aggregate
from shb.agents import LLMAgent

def complete(prompt: str) -> str:
    ...  # call your model; return its text

agent = LLMAgent(complete, name="my-model")
grades = run_benchmark(agent, seeds=range(10))
print(aggregate("my-model", grades).to_markdown())
```

The adapter renders each task (prompt + files) and asks for a strict-JSON answer with a
first-class `issues_detected` / `abstain` channel, so the model can flag a flaw it spots —
which is exactly what the trapped tasks reward. See [`shb/agents/llm.py`](shb/agents/llm.py).

## Coverage

**8 domains · 26 task families · all 12 trap types.** Each family is grounded in a real
library and built so the careful reference passes and the naive reference fails.

| domain | families | example traps |
|---|---|---|
| chemistry | 4 | decoy molar mass, corrupt SMILES, kcal/kJ unit mismatch, QM non-convergence |
| statistics | 4 | target leakage, Simpson's paradox, multiple comparisons, underpowered n |
| physics | 3 | CFL instability, unit mismatch, out-of-domain extrapolation |
| biology | 3 | batch confounding, wrong control, corrupt FASTA |
| climate | 3 | future leakage, Kelvin/Celsius mismatch, decoy station |
| astronomy | 3 | corrupt pixels, non-converged fit, magnitude/flux mismatch |
| neuroscience | 3 | circular analysis (double dipping), multiple comparisons, underpowered |
| materials | 3 | non-physical crystal, eV/meV mismatch, composition extrapolation |

The full failure-mode taxonomy is in [docs/TAXONOMY.md](docs/TAXONOMY.md).

## How it fits together

```
shb/
  taxonomy.py      # the 12 fake-science TrapTypes, with literature grounding
  types.py         # the contract: TaskInstance, AgentView, Submission, Grade, Family
  utils.py         # standard_grade — the clean/trapped policy
  registry.py      # families self-register on import
  runner.py        # generate -> hand the agent a public view -> grade
  scoring.py       # competence, robustness, fake-science gap, trap-detection rate
  agents/          # ReferenceAgent (naive/careful) + LLMAgent adapter
  domains/         # one module per domain; each registers its families
scripts/           # run_benchmark.py, make_scorecard.py
tests/             # contract, scoring, harness, and the discrimination invariant
results/           # the reference scorecard artifact
docs/              # SPEC, TAXONOMY, METRICS, CONTRIBUTING, LEADERBOARD
```

Adding a domain or family is ~100 lines against a fixed contract — see
[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md). The design and grading details are in
[docs/SPEC.md](docs/SPEC.md) and [docs/METRICS.md](docs/METRICS.md). Submission and the
public-dev / private-test split policy are in [docs/LEADERBOARD.md](docs/LEADERBOARD.md).

## Honest limitations

- **Tasks are procedurally generated, grounded in real computation** — not lifted verbatim
  from specific papers. The *methods* and *failure modes* come from the literature (the
  reproducibility crisis, known numerical pitfalls); the instances are synthesized so they
  can be regenerated and graded without a human. A literature-curated clean-task track is
  natural future work.
- **The reference agents are oracles, not contestants.** `reference-careful` encodes the
  correct method per family; it exists to prove the benchmark is solvable and discriminating,
  and to document what "doing it right" means. The systems under test are real agents that do
  not have these solvers.
- **One trap per trapped instance.** Real analyses fail in combinations; v1 isolates single
  failure modes so the gap attributes cleanly to a named trap.
- **Detection is keyword/abstention based.** A trapped task passes if the agent delivers the
  corrected answer, abstains, or flags the planted flaw in its `issues_detected` channel.

## License

MIT — see [LICENSE](LICENSE).
