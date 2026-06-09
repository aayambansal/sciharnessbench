# Submission & the contamination policy

SciHarnessBench is a **public benchmark**, which makes contamination the central threat: the
moment a fixed set of tasks (and their answers) is published, the next model can train on it.
Because the suite is **self-generating**, we defend against this structurally rather than
hoping the test set stays secret. The full threat model and the isolated official protocol are
in [THREAT_MODEL.md](THREAT_MODEL.md); this page is the submission policy.

## The public-dev / private-test split

- **A dev split is public.** The salt-free instances (a held-out dev seed range) ship in the
  repo with their graders. Use them to develop, debug, and self-report. Memorizing them buys
  nothing on the official score.
- **The scored test split is private and salted.** A secret `SHB_SALT` mixes into every
  generator (`utils.family_rng` / `utils.np_seed`) and into the opaque task ids
  (`types.opaque_id`), and the official run uses a held-out seed range. Same families, same
  code, fresh instances. A model that memorized the public dev instances gains nothing — it
  must actually validate inputs.
- **Isolated protocol.** The official run executes the agent **out-of-process** with only
  serialized public assets; the `shb` package, generators, graders, and reference solvers are
  not present in that environment, so an agent cannot call the careful oracle or read hidden
  fields. Per-task time/memory limits apply; a timeout or crash is an unsafe failure (score 0,
  not confident-wrong). See [THREAT_MODEL.md](THREAT_MODEL.md).

Because instances are regenerable, contamination is repaired by a **versioned refresh** —
bumping the salt / seed manifest to a `v1.1`, `v2`, … — rather than rebuilding the benchmark.
The families and code are unchanged; only the instances refresh. This is the SWE-bench
"Verified/Live" idea made automatic.

## How to submit

1. Implement an agent: wrap your model/agent loop in `shb.agents.LLMAgent` (a
   `complete(prompt) -> str` callable) or any object with `solve(view, workdir) -> Submission`.
   The agent returns structured `issues` (`{"kind", "evidence"}`, kinds drawn from
   `view.allowed_issue_kinds`) and a `confidence`.
2. Self-report on the public dev split:
   ```bash
   python scripts/run_benchmark.py --seeds 10   # or run your LLMAgent via run_benchmark()
   ```
3. Open a PR / submission adding your agent adapter and your dev-split scorecard JSON. The
   maintainers run the identical agent on the private, salted test split under the isolated
   protocol and publish the official numbers.

## Rules

- **Do not train on, fine-tune on, or hard-code the suite** (families, seeds, or answers). The
  benchmark measures input-validation behavior, not recall of these tasks. Submissions are
  audited for benchmark-specific hard-coding.
- **Report the salt/seed manifest and version** used for any number you publish.
- **Report competence AND robustness AND confident-wrong, never robustness alone.** A high
  robustness with low competence (e.g. an always-abstain agent) is not a result — and because
  false alarms fail clean tasks, it costs competence anyway. See [METRICS.md](METRICS.md).
- **The reference agents are not contestants.** `reference-naive` and `reference-careful` are
  oracles that bound the achievable range and document correct vs fake-science behavior; they
  are not leaderboard entries.

## What gets reported

For each agent: **competence**, **robustness**, the **fake-science gap** with its paired
bootstrap 95% CI, **confident-wrong rate**, **false-alarm rate**, and **trap-detection rate**,
in both micro and macro aggregations, plus the breakdowns **by domain** and **by trap type**
(so a system's specific blind spots — say, unit mismatches or confounding — are visible, not
averaged away). Non-paired families are reported as separate robustness scenarios. Optionally:
cost / tokens / wall-clock per task, reported alongside but never folded into the correctness
metrics.
