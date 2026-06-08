# Submission & the contamination policy

SciHarnessBench is a **public benchmark**, which makes contamination the central threat: the
moment a fixed set of tasks (and their answers) is published, the next model can train on it.
Because the suite is **self-generating**, we defend against this structurally rather than
hoping the test set stays secret.

## The public-dev / private-test split

- **Clean tasks are fully public.** Clean-task answers may well be memorized — we don't care,
  because clean accuracy (competence) is not the headline. The headline is the gap, which is
  driven by the trapped tasks.
- **A dev split is public:** seeds `0–9` of every family ship in the repo, with their graders.
  Use them to develop, debug, and self-report.
- **The scored test split is private and regenerable:** a held-out seed range (and, per
  release, a secret salt mixed into the family seed strings) that only the maintainers run.
  Same families, same code, fresh instances. A model that memorized the public dev instances
  gains nothing on the private split — it must actually validate inputs.

Because instances are regenerable, contamination is repaired by **bumping the salt / seed
range** (a `v1.1`, `v2`, …) rather than rebuilding the benchmark. This is the SWE-bench
"Verified/Live" idea made automatic.

## How to submit

1. Implement an agent: wrap your model/agent loop in `shb.agents.LLMAgent` (a
   `complete(prompt)->str` callable) or any object with `solve(view, workdir) -> Submission`.
2. Self-report on the public dev split:
   ```bash
   python scripts/run_benchmark.py --seeds 10   # or run your LLMAgent via run_benchmark()
   ```
3. Open a PR / submission adding your agent adapter and your dev-split scorecard JSON. The
   maintainers run the identical agent on the private test split and publish the official
   numbers.

## Rules

- **Do not train on, fine-tune on, or hard-code the suite** (families, seeds, or answers).
  The benchmark measures input-validation behavior, not recall of these tasks.
- **Report the seed range and version** used for any number you publish.
- **Report competence and robustness, never robustness alone.** A high robustness with low
  competence (e.g. an always-abstain agent) is not a result; see [METRICS.md](METRICS.md).
- **The reference agents are not contestants.** `reference-careful` is an oracle that encodes
  the correct method per family; it bounds the scale and is not a leaderboard entry.

## What gets reported

For each agent: overall **competence**, **robustness**, **fake-science gap**, and **trap
detection rate**, plus the same broken down **by domain** and **by trap type** (so a system's
specific blind spots — say, unit mismatches or confounding — are visible, not averaged away).
Optionally: cost / tokens / wall-clock per task, reported alongside but never folded into the
correctness metrics.
