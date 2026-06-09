# Threat model and the official evaluation protocol

A benchmark score is only meaningful if the system under test cannot obtain it by
gaming. SciHarnessBench assumes the agent may be adversarial: it may inspect the
task, exploit ordering, stuff flags, or try to reach benchmark internals. This
document lists each channel and how it is closed, and specifies the isolated,
salted protocol used for official scores.

## Channels and closures

| # | Channel | Closure | Enforced by |
|---|---------|---------|-------------|
| 1 | **Variant label leak** — read `clean`/`trapped` (or the seed) off the task id or file path | The public `task_id` is an opaque salted hash; no variant/seed/twin info in the view or the workdir path | `types.opaque_id`, `runner._write_assets`; `tests/test_contract`, `tests/test_adversarial` |
| 2 | **Twin-order exploitation** — a stateful agent compares the clean and trapped twin | Instances are presented in a deterministically shuffled order with twins never adjacent | `runner._ordered` |
| 3 | **Keyword stuffing** — dump trap vocabulary into free text | Detection requires a **structured** issue of the **correct kind** carrying **evidence the grader verifies** against ground truth (offending ids, units, residuals). Free-text notes are ignored for detection | `utils.standard_grade`, each family's `evidence_ok`; `tests/test_scoring`, `tests/test_adversarial` |
| 4 | **Flag/abstain everything** — claim a flaw or abstain on every task | A flaw claim or abstention on a **clean** task is a false alarm scoring zero, so this collapses competence; the headline requires high competence *and* robustness | `utils.standard_grade`; `tests/test_adversarial` (stuffer/abstain agents score ~0 competence) |
| 5 | **Reach benchmark internals** — `import shb` and call the careful oracle or a generator | Official evaluation runs the agent **out-of-process** with only serialized public assets; `shb` is not importable in the agent environment | Official protocol below (the in-process runner is for the reference oracles and local dev only) |
| 6 | **Memorize public instances** | A **private salted split**: `SHB_SALT` mixes into every generator (`family_rng`/`np_seed`), so the official test set uses a held-out salt and seed range unknown to submitters; releases are versioned and refreshable | `utils.family_rng`, `utils.np_seed`, `types.opaque_id` |
| 7 | **Hard-code family parsers** | Held-out templates/families plus the salt of (6); the public set is a transparent dev split, and submissions are audited for benchmark-specific hard-coding | Official protocol + release policy |

## What the in-process numbers mean

The public (salt-free) instances and the in-process `run_benchmark` are a
**transparent development split**: anyone can reproduce the reference scorecard
and debug an agent. They are not the official score. A trustworthy leaderboard
number comes from the isolated, salted run below.

## Official evaluation protocol

1. **Isolation.** The agent runs in a separate process (or container) and
   receives each task as serialized public assets over a task-serving API. The
   `shb` package, registry, generators, graders, and reference solvers are not
   present in that environment. Per-task wall-clock and memory limits apply; a
   timeout or crash is recorded as an unsafe failure (score zero, not
   confident-wrong).
2. **Private salt + seed range.** The official set is generated with a secret
   `SHB_SALT` and a held-out seed range. Memorizing the public dev instances
   yields no advantage, because the salted instances differ.
3. **Versioned refresh.** Each release pins a salt/seed manifest. When
   contamination is suspected, the maintainers rotate the salt (a `vN+1`
   release) — the families and code are unchanged; only the instances refresh.
4. **Submission audit.** Submitted agents are reviewed for hard-coding of
   family-specific answers or parsers that would not generalize to held-out
   templates.

The reference implementation ships the salt mechanism (`SHB_SALT`) and the
deterministic, opaque-id, shuffled in-process runner; the production task-serving
server is the deployment of this protocol.
