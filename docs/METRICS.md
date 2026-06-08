# Metrics

Defined in `shb/scoring.py`; the per-task policy is in `shb/utils.py::standard_grade`.

## Per-task scoring

Each instance gets a 0/1 `score` and (for trapped tasks) a `trap_detected` flag, under one
policy:

| variant | the agent... | outcome |
|---|---|---|
| clean | delivers the correct answer | **pass** (score 1) |
| clean | abstains or flags a (nonexistent) flaw | **fail** — false alarm (score 0) |
| trapped | delivers the trap-corrected answer | **pass** |
| trapped | abstains, or flags the planted flaw (keyword match) | **pass**, `trap_detected = true` |
| trapped | confidently reports the naive wrong answer | **fail** (score 0) — the fake science |
| any | raises an exception | **fail** (score 0) |

A trapped task counts as detected if the agent abstains or its `issues_detected`/`notes`
contain one of the family's trap keywords (case-insensitive).

## Aggregate metrics

- **Competence (C)** = mean score on clean tasks. *Can the agent do the science?*
- **Robustness (R)** = mean score on trapped tasks. *Does it hold up when the data is flawed?*
- **Fake-Science Gap (FSG) = C − R** — the headline. ~0 means the agent is as reliable on
  flawed data as on clean data; a large positive gap means it does fake science.
- **Trap Detection Rate (TDR)** = fraction of trapped tasks where the planted flaw was
  explicitly flagged. (An agent can pass by delivering the corrected answer without flagging,
  so TDR ≤ R in general; TDR rewards *explicit* skepticism.)

All four are reported overall and broken down **by domain**, **by family**, and **by trap
type** (`Scorecard.to_markdown()` / `to_json()`).

## Why the gap is the right headline

1. **Within-task difference, not an absolute judge score.** C and R are measured on the same
   tasks; their difference cancels a lot of task-difficulty and grading noise, and it does not
   drift as the underlying judge improves (there is no LLM judge).
2. **Abstention-proof.** Because false alarms fail clean tasks, you cannot inflate R by
   abstaining everywhere — that destroys C. The only way to a low FSG is to actually be
   competent *and* robust. Report C and R together; FSG alone (e.g. a negative gap from an
   always-abstain agent) is meaningless without C.
3. **Crashes count.** Refusing to validate a corrupt input and throwing is a failure, not an
   exemption.

## Reference bounds

The two reference agents bracket the scale (50 seeds × 26 families):

| agent | C | R | FSG | TDR |
|---|---:|---:|---:|---:|
| `reference-naive` | 100% | 0% | 100 pts | 0% |
| `reference-careful` | 100% | 100% | 0 pts | 100% |

A system under test should be read against these: high C near the careful line is table
stakes; the interesting axis is how far its R falls below C — i.e. how much of its apparent
competence is fake science.
