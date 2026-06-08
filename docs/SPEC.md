# SciHarnessBench specification

This is the contract every task family, agent, and the harness share. It is enforced by
`shb/types.py` and exercised by `tests/test_contract.py`.

## Roles

- **Task family** — generates problem instances and grades submissions. One per scientific
  task type (e.g. `chem.theoretical_yield`).
- **Agent** — the system under test. Receives only a public `AgentView`; returns a
  `Submission`.
- **Harness** — generates instances, writes their assets to a scratch dir, hands the agent a
  view, and calls the family's grader.

## A task instance

`TaskInstance` (see `shb/types.py`) is one concrete problem — either a clean task or its
trapped twin.

| field | visible to agent? | meaning |
|---|---|---|
| `task_id` | yes | `family/seed=N/variant` |
| `family`, `domain` | yes | identifiers |
| `prompt` | yes | natural-language task statement |
| `assets` | yes | `{filename: text}`, written into the agent's workdir |
| `answer_fields` | yes | `{field: description}` the agent must return |
| `variant` | **no** | `clean` or `trapped` |
| `trap_type`, `trap_note` | **no** | which flaw was planted, and a description |
| `answer` | **no** | hidden ground truth |
| `grading` | **no** | hidden grader payload (tolerances, trap keywords) |

The agent only ever sees `inst.view()`, an `AgentView` carrying the visible fields. The
hidden fields are not attributes of the view at all (`tests/test_contract.py` asserts this).

## What an agent returns

`Submission`:

- `answers: dict` — the requested `answer_fields`.
- `issues_detected: list[str]` — the standard channel for flagging a planted flaw
  ("molar mass disagrees with structure", "SCF did not converge"). Graders match these
  against the family's trap keywords.
- `abstained: bool` — the agent judges the task unanswerable as posed. A legitimate, correct
  response to many traps.
- `notes: str` — free text (also searched for trap keywords).

## What a grader returns

`Grade`: `conclusion_correct` (final answer scientifically acceptable), `trap_detected`
(trapped only — was the planted flaw flagged?), `score` (0/1), and `detail`.

## The family contract

```python
Family(
    family_id, domain, title, description, trap_types,
    generate,     # (seed: int, variant: str) -> TaskInstance
    grade,        # (inst: TaskInstance, sub: Submission) -> Grade
    ref_naive,    # (view: AgentView) -> Submission   (trusts inputs -> fake science)
    ref_careful,  # (view: AgentView) -> Submission   (validates -> real science)
)
```

Rules:

1. **Determinism.** `generate(seed, variant)` must be a pure function of `(seed, variant)`.
   Seed a base RNG with `f"{FAMILY}:{seed}"` and a separate trap RNG with
   `f"{FAMILY}:{seed}:trap"`. Do **not** seed numpy via
   `int.from_bytes(s.encode(), "little") % 2**32` — for prefix-sharing strings that collapses
   every seed to one problem. Use `random.Random(str)` or `hashlib.sha256(s.encode())`.
2. **Twins differ only by the trap.** For a given seed, the clean and trapped instances must
   share the same base problem and differ only by the injected flaw, so the gap attributes
   cleanly to the trap.
3. **Self-consistency.** Ground truth is computed at `generate()` time with a real library.
   The clean instance must be solvable to that ground truth (a `CONVERGED` log must really
   end below tolerance; a "significant effect" clean task must actually be significant).
4. **Reference baselines read only the view.** `ref_naive` and `ref_careful` parse
   `view.assets` like a real agent; they never touch hidden fields. `ref_careful` must NOT
   false-alarm on clean tasks.

## The run loop

For each selected family, seed, and variant (`shb/runner.py`):

1. `inst = family.generate(seed, variant)`
2. write `inst.assets` into a per-instance workdir
3. `sub = agent.solve(inst.view(), workdir)` — if this raises, the task scores 0
4. `grade = family.grade(inst, sub)`

Grades are aggregated by `shb/scoring.py` into a scorecard. See [METRICS.md](METRICS.md).
