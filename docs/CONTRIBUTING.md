# Contributing a task family or domain

A family is ~60–100 lines against the fixed v2 contract. The canonical reference is
`shb/domains/chemistry.py` — copy its structure. Read [SPEC.md](SPEC.md) first.

## Recipe

A family is four functions plus a `Family` registration. Mirror `chemistry.py`:

```python
from ..taxonomy import TrapType
from ..types import Family, Submission, TaskInstance
from ..utils import (base_signature, family_rng, np_seed, issue,
                     standard_grade, ev_text, ev_near, ev_contains)

FAM = "mydomain.my_task"

def my_generate(seed, variant):
    rng = family_rng(FAM, seed)                     # salted base RNG (same for both twins)
    # ... build the base problem from rng; compute ground truth with a REAL library ...
    trap_type, trap_note = None, ""
    grading = {"base_signature": base_signature(<the base, NOT the trap>),
               "prompt_uncued": "<task statement with NO trap vocabulary>"}
    if variant == "trapped":
        t = family_rng(FAM, seed, trap=True)        # independent salted trap RNG
        # ... inject exactly one flaw so the correct answer changes ...
        # ... stash whatever evidence_ok needs to verify a detection (ids, true value) ...
        trap_type, trap_note = TrapType.SOMETHING, "what was planted"
    return TaskInstance(
        task_id=f"{FAM}/seed={seed}/{variant}", family=FAM, domain="mydomain",
        variant=variant, seed=seed,
        prompt="<cued statement: ask the agent to validate inputs>",
        assets={...}, answer_fields={...},
        trap_type=trap_type, trap_note=trap_note,
        answer={...},                               # HIDDEN ground truth
        grading=grading)

def my_grade(inst, sub):
    correct = ...                                   # compare sub.answers to inst.answer for THIS variant

    def ev_ok(ev):                                  # verify a detection's evidence vs ground truth
        return inst.grading["..."] and ev_near(ev, inst.grading["true_value"])

    return standard_grade(inst, sub, answer_correct=correct,
                          flaw_kind="something", evidence_ok=ev_ok)

def _solve(view, careful):
    # parse view.assets like a real agent. If careful: validate, correct the answer,
    # AND emit a structured, evidence-bearing issue. If naive: trust inputs, answer confidently.
    issues = [issue("something", offending_id=..., true_value=...)] if (careful and flawed) else []
    return Submission(answers={...}, issues=issues, confidence=0.9 if careful else 1.0)

registry.register(Family(
    FAM, "mydomain", "Title", "one-line description",
    [TrapType.SOMETHING],          # trap_types
    "something",                   # flaw_kind == the TrapType value a detection must name
    my_generate, my_grade,
    lambda v: _solve(v, False), lambda v: _solve(v, True),
    paired=True))                  # False only if the twins differ by more than the injected flaw
```

Then add your module name to `DOMAIN_MODULES` in `shb/domains/__init__.py` (a new domain) or
just add the family to an existing domain module.

## The rules that actually matter

1. **Salted determinism.** `generate(seed, variant)` is a pure function of its args. Seed with
   `family_rng(FAM, seed)` (base), `family_rng(FAM, seed, trap=True)` (trap), and numpy via
   `np_seed(...)`. **Do not** hand-roll `int.from_bytes(s.encode(), ...) % 2**32` — prefix-
   sharing strings collapse to one problem. The helpers mix in `SHB_SALT` for you.
2. **Twins differ only by the trap (`paired=True`).** Build the base from the base RNG; apply
   the trap from the trap RNG. Store `grading["base_signature"] = base_signature(<base>)` so
   both twins share it — `tests/test_contract.py` checks twins match and that seeds differ. If
   the clean and trapped variants differ in the underlying signal or sample (not a pure
   injection), set `paired=False`; the family is then reported as a robustness scenario, never
   in the gap.
3. **`flaw_kind` is in the vocabulary.** It must be a `TrapType` value (see
   `utils.allowed_issue_kinds()`); `grade` passes it to `standard_grade`, and `ref_careful`
   uses the same string in its `issue(...)`.
4. **Evidence that verifies against ground truth.** `evidence_ok` must check the issue's
   `evidence` against the hidden truth (the offending ids, the detected unit, the residual) —
   not just look for a keyword. Keyword stuffing must fail. Stash whatever it needs in
   `grading` at generate time.
5. **The trap must bite.** The injected flaw must change the correct answer beyond the grader's
   tolerance, so `ref_naive` actually fails trapped.
6. **The clean twin must be self-consistent.** Its ground truth must be derivable from its own
   assets — a "converged" log ends below tolerance; a "significant" clean result is actually
   significant. Otherwise `ref_careful` false-alarms or fails clean and competence drops.
7. **Cued + uncued prompts.** The default `prompt` is cued (it asks the agent to validate
   inputs). Provide `grading["prompt_uncued"]` with no trap vocabulary for the uncued track.
8. **Reference solvers read only the view.** `ref_careful` must **not** false-alarm on clean
   tasks; `ref_naive` must fail trapped tasks confidently. They double as the proof your family
   discriminates and the documentation of correct vs fake-science behavior.

## Verify before you open a PR

```bash
python scripts/run_benchmark.py --agent both --domains mydomain --seeds 10
```

Required:

- `reference-careful` competence **and** robustness ~100% (gap ~0) with **false-alarm 0%**;
- `reference-naive` competence ~100% with robustness **low** (gap large).

Then:

```bash
pytest -q     # contract + scoring + discrimination + adversarial invariants must stay green
```

## Checklist

- [ ] New `TrapType` (if any) added to `shb/taxonomy.py` with a `TRAP_META` entry.
- [ ] `flaw_kind` is one of `allowed_issue_kinds()` and matches the issue kind `ref_careful` emits.
- [ ] Ground truth computed by a real library, stored in `answer`, hidden from the view.
- [ ] `evidence_ok` verifies evidence against ground truth (not a keyword); keyword stuffing fails.
- [ ] Paired twins share `grading["base_signature"]`; non-twins set `paired=False`.
- [ ] Cued `prompt` plus a `grading["prompt_uncued"]` with no trap vocabulary.
- [ ] `--domains mydomain --seeds 10`: careful 100/100 with false-alarm 0; naive 100/low.
- [ ] `pytest -q` green; the family added to `docs/TAXONOMY.md` coverage.
- [ ] Only your domain module changed; no edits to core (`types`, `utils`, `runner`, `scoring`).
