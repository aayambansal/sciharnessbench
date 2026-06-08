# Contributing a task family or domain

A family is ~60–100 lines against a fixed contract. The best reference is
`shb/domains/chemistry.py`; copy its structure. Read [SPEC.md](SPEC.md) first.

## Recipe

A family is four functions plus a `Family` registration:

```python
FAM = "mydomain.my_task"

def my_generate(seed, variant):
    rng  = random.Random(f"{FAM}:{seed}")          # base RNG
    # ... build the base problem from rng (same for clean and trapped) ...
    # ... compute ground truth with a REAL library; store in answer ...
    trap_type, trap_note = None, ""
    if variant == "trapped":
        t = random.Random(f"{FAM}:{seed}:trap")     # separate trap RNG
        # ... inject exactly one flaw so the correct answer changes ...
        trap_type, trap_note = TrapType.SOMETHING, "what was planted"
    return TaskInstance(task_id=f"{FAM}/seed={seed}/{variant}", family=FAM,
                        domain="mydomain", variant=variant, seed=seed,
                        prompt=..., assets={...}, answer_fields={...},
                        trap_type=trap_type, trap_note=trap_note,
                        answer={...}, grading={"trap_keywords": [...]})

def my_grade(inst, sub):
    correct = ...  # compare sub.answers to inst.answer
    return standard_grade(inst, sub, correct_now=correct,
                          trap_keywords=inst.grading["trap_keywords"])

def _solve(view, careful):
    # parse view.assets like a real agent; if careful, validate + flag/correct
    ...
    return Submission(answers={...}, issues_detected=[...] if careful else [])

registry.register(Family(FAM, "mydomain", "Title", "one-line description",
    [TrapType.SOMETHING], my_generate, my_grade,
    lambda v: _solve(v, False), lambda v: _solve(v, True)))
```

Then add your module name to `DOMAIN_MODULES` in `shb/domains/__init__.py` (a new domain) or
just add the family to an existing domain module.

## The five rules that actually matter

1. **Determinism.** `generate(seed, variant)` is a pure function of its args. Seed with
   `f"{FAM}:{seed}"` and `f"{FAM}:{seed}:trap"`. **Do not** seed numpy with
   `int.from_bytes(s.encode(), "little") % 2**32` — prefix-sharing strings collapse to one
   problem. Use `random.Random(str)` or `hashlib.sha256(s.encode()).digest()[:8]`.
2. **Twins differ only by the trap.** Build the base from the base RNG; apply the trap from
   the trap RNG. `tests/test_contract.py` checks determinism and distinct-per-seed problems.
3. **The trap must bite.** The injected flaw must change the correct answer beyond the
   grader's tolerance, so the naive baseline actually fails. (If it doesn't, target the flaw
   at the quantity that matters, or verify-and-resample at generate time.)
4. **The clean twin must be self-consistent.** Its ground truth must be derivable from its
   own assets — a "converged" log ends below tolerance; a "significant" clean result is
   actually significant. Otherwise the careful baseline fails clean and competence drops.
5. **`ref_careful` must not false-alarm on clean tasks**, and `ref_naive` must fail trapped
   tasks. These two reference solvers double as the proof your family discriminates and the
   documentation of correct vs fake-science behavior.

## Verify before you open a PR

```bash
python scripts/run_benchmark.py --agent both --domains mydomain --seeds 5
```

Required: `reference-careful` competence **and** robustness ~100% (gap ~0);
`reference-naive` competence ~100% with robustness **low** (gap large). Then:

```bash
pytest -q     # contract + discrimination invariants must stay green
```

## Checklist

- [ ] New `TrapType` (if any) added to `shb/taxonomy.py` with a `TRAP_META` entry.
- [ ] Ground truth computed by a real library, stored in `answer`, hidden from the view.
- [ ] Trap keywords listed in `grading["trap_keywords"]`.
- [ ] `--domains mydomain --seeds 5` shows the careful≫naive pattern.
- [ ] `pytest -q` green; the family added to `docs/TAXONOMY.md` coverage if it adds a trap.
- [ ] Only your domain module changed; no edits to core (`types`, `utils`, `runner`, `scoring`).
