# SciHarnessBench specification

This is the v2 contract every task family, agent, and the harness share. It is enforced by
`shb/types.py` and exercised by `tests/test_contract.py`. The design closes the gaming
channels in [THREAT_MODEL.md](THREAT_MODEL.md): the agent sees an **opaque** view, reports
flaws through a **structured, evidence-bearing** channel, and reports a **confidence**.

## Roles

- **Task family** — generates problem instances and grades submissions. One per scientific
  task type (e.g. `chem.theoretical_yield`).
- **Agent** — the system under test. Receives only a public `AgentView`; returns a
  `Submission`.
- **Harness** — generates instances, writes their assets to an opaque scratch dir, hands the
  agent a view, and calls the family's grader.

## A task instance

`TaskInstance` (see `shb/types.py`) is one concrete problem — either a clean task or its
trapped twin. The internal `task_id` is `family/seed=N/variant`; the agent never sees it.

| field | visible to agent? | meaning |
|---|---|---|
| `task_id` (internal) | **no** | `family/seed=N/variant` — used only in our reports |
| `domain`, `family` | yes | identifiers; `family` is the task *type* and does not reveal the trap |
| `prompt` | yes | natural-language task statement |
| `assets` | yes | `{filename: text}`, written into the agent's workdir |
| `answer_fields` | yes | `{field: description}` the agent must return |
| `variant` | **no** | `clean` or `trapped` |
| `seed` | **no** | the (salted) instance seed |
| `trap_type`, `trap_note` | **no** | which flaw was planted, and a description |
| `answer` | **no** | hidden ground truth |
| `grading` | **no** | hidden grader payload (tolerances, offending ids, `base_signature`, `prompt_uncued`) |

## The opaque agent view

The agent only ever sees `inst.view(prompt_style)`, an `AgentView` carrying:

| field | meaning |
|---|---|
| `task_id` | an **opaque salted hash** (`types.opaque_id`) — carries no variant, seed, or twin info |
| `domain`, `family` | the domain and task type |
| `prompt` | the task statement (cued by default; the uncued variant if `prompt_style="uncued"`) |
| `assets` | `{filename: text}` |
| `answer_fields` | the fields to return |
| `allowed_issue_kinds` | the published controlled vocabulary (the 12 `TrapType` values) for structured flaw reports |

The variant, seed, ground-truth answer, and grading payload are not attributes of the view at
all (`tests/test_contract.py` asserts this). Mixing `SHB_SALT` into `opaque_id` means
memorizing public ids buys nothing on the official split. The agent must discover any planted
flaw from the data alone, and may flag it only with a kind drawn from `allowed_issue_kinds`.

## What an agent returns

`Submission`:

- `answers: dict` — the requested `answer_fields`.
- `issues: list[dict]` — the **structured** flaw channel. Each entry is
  `{"kind": <one of allowed_issue_kinds>, "evidence": <verifiable detail>}`. To get credit for
  catching a trap, the agent must name the **correct kind** *and* supply **evidence the grader
  verifies** against ground truth (the offending ids, the detected unit, the non-converged
  residual). There is no free-text `issues_detected` channel, and `notes` is **not** searched
  for keywords — keyword stuffing does not count. Use `utils.issue(kind, **evidence)` to build
  one.
- `confidence: float` in `[0,1]` — the agent's confidence in its `answers`. It powers the
  confident-wrong-rate metric.
- `abstained: bool` — the agent judges the task unanswerable as posed. On a trapped task a
  bare abstention does **not** pass; an abstention carrying a verified structured issue does.
- `notes: str` — free text, recorded but **ignored for grading**.

## What a grader returns

`Grade` (one per instance/submission pair):

| field | meaning |
|---|---|
| `conclusion_correct` | the answer is scientifically acceptable (corrected, or a valid detection) |
| `answer_correct` | the *answer field* matches, ignoring detection |
| `trap_detected` | trapped only: planted flaw flagged with valid evidence (else `None`) |
| `false_alarm` | a flaw claim or abstention on a **clean** task |
| `confident_wrong` | trapped: wrong answer, undetected, asserted with `confidence >= 0.5` |
| `confidence`, `score` | the reported confidence and the 0/1 score |
| `paired`, `trap_type` | whether this family is a paired twin; the trap value |
| `detail` | per-grade diagnostics (and `error` if the agent crashed) |

## The family contract

```python
Family(
    family_id, domain, title, description,
    trap_types,   # list[TrapType]
    flaw_kind,    # the controlled-vocabulary kind a correct detection must name
    generate,     # (seed: int, variant: str) -> TaskInstance
    grade,        # (inst: TaskInstance, sub: Submission) -> Grade
    ref_naive,    # (view: AgentView) -> Submission   (trusts inputs -> fake science)
    ref_careful,  # (view: AgentView) -> Submission   (validates -> real science)
    paired=True,  # True only for genuine counterfactual twins
)
```

Rules:

1. **Salted determinism.** `generate(seed, variant)` is a pure function of `(seed, variant)`.
   Seed RNGs with `utils.family_rng(family_id, seed)` (base) and
   `family_rng(family_id, seed, trap=True)` (trap), and numpy via `utils.np_seed(...)`. Both
   mix in `SHB_SALT`, so the salt-free public instances are a transparent dev split and the
   official set uses a held-out salt (see [THREAT_MODEL.md](THREAT_MODEL.md)).
2. **Twins differ only by the trap (`paired=True`).** For a seed, the clean and trapped
   instances must be identical except the injected flaw, and store a shared
   `grading["base_signature"]` (`utils.base_signature(...)`) so a test can verify it. A family
   whose clean/trapped variants differ in the underlying signal or sample (not a pure
   injection) sets `paired=False` and is reported as a **robustness scenario**, never folded
   into the paired Fake-Science Gap.
3. **`flaw_kind` is in the vocabulary.** It is the `TrapType` value a correct detection must
   name; `grade` passes it to `standard_grade`, and the agent must use it in
   `Submission.issues`.
4. **Self-consistency.** Ground truth is computed at `generate()` time with a real library and
   stored in the hidden `answer`. The clean instance must be solvable to that ground truth (a
   `CONVERGED` log really ends below tolerance; a "significant" clean result is significant).
5. **Cued and uncued prompts.** The default `prompt` is the cued track (it asks the agent to
   validate inputs). Provide a `grading["prompt_uncued"]` variant with no trap vocabulary for
   the uncued track; `view("uncued")` swaps it in.
6. **Reference baselines read only the view.** `ref_naive` and `ref_careful` parse
   `view.assets` like a real agent and never touch hidden fields. `ref_naive` answers
   confidently (commits fake science); `ref_careful` validates, corrects, and emits a
   structured, evidence-bearing issue — and must **not** false-alarm on clean tasks.

## The grading policy

Graders call `utils.standard_grade`, which applies the v2 clean/trapped policy:

```python
standard_grade(
    inst, sub, *,
    answer_correct: bool,                 # do the answer fields match the correct value for THIS instance?
    flaw_kind: str,                       # the kind a correct detection must name
    evidence_ok: Callable[[Any], bool],   # verifies an issue's evidence against ground truth
    detail: dict | None = None,
) -> Grade
```

- **Clean task:** pass only by the correct answer with **no** issue and **no** abstention. Any
  issue or abstention is a *false alarm* (score 0).
- **Trapped task:** pass by the trap-corrected answer **or** by a verified structured detection
  — an `issues_of(flaw_kind)` entry whose evidence satisfies `evidence_ok`. A wrong, undetected
  answer asserted with `confidence >= 0.5` is a *confident-wrong* event.

A crash is caught by the runner, scored 0 as an **unsafe failure** (recorded in `detail.error`),
and is **not** counted as confident-wrong. See [METRICS.md](METRICS.md) for the full policy
table and aggregates.

## The run loop

For each selected family, seed, and variant (`shb/runner.py`):

1. `inst = family.generate(seed, variant)`
2. write `inst.assets` into a per-instance workdir whose name is the **opaque** `public_id`
   (no variant/seed in the path)
3. `sub = agent.solve(inst.view(prompt_style), workdir)` — if this raises, the task is scored 0
   as an unsafe failure
4. `grade = family.grade(inst, sub)`

Instances are presented in a **deterministically shuffled** order with clean/trapped twins
never adjacent (`_ordered`), so a stateful agent cannot exploit ordering. `prompt_style`
selects the cued (default) or uncued track. Grades are aggregated by `shb/scoring.py` into a
scorecard. See [METRICS.md](METRICS.md) and [THREAT_MODEL.md](THREAT_MODEL.md).
