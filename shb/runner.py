"""Run an agent over task families and collect grades.

For each (family, seed, variant) the runner generates an instance, writes its
assets into an **opaque** scratch dir (no variant/seed in the path), hands the
agent only the public view, and grades the result. Instances are presented in a
**deterministically shuffled** order with clean/trapped twins never adjacent, so
a stateful agent cannot exploit ordering. An agent that raises is scored as a
failed conclusion (not as "confident wrong" — a crash is an unsafe failure, a
distinct outcome we record separately).
"""
from __future__ import annotations

import os
import random
import tempfile
import traceback
from typing import Iterable, Optional

from . import registry
from .types import Agent, Family, Grade, TaskInstance


def _write_assets(inst: TaskInstance, root: str) -> str:
    d = os.path.join(root, inst.public_id)          # opaque dir name
    os.makedirs(d, exist_ok=True)
    for name, content in inst.assets.items():
        with open(os.path.join(d, name), "w") as fh:
            fh.write(content)
    return d


def _error_grade(inst: TaskInstance, fam: Family, exc: Exception) -> Grade:
    return Grade(
        task_id=inst.task_id, family=inst.family, domain=inst.domain, variant=inst.variant,
        seed=inst.seed, trap_type=inst.trap_type.value if inst.trap_type else None, paired=fam.paired,
        conclusion_correct=False, answer_correct=False,
        trap_detected=False if inst.is_trapped else None,
        false_alarm=False, confident_wrong=False, confidence=0.0, score=0.0,
        detail={"error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-3:]})


def run_instance(agent: Agent, fam: Family, inst: TaskInstance, root: str,
                 prompt_style: str = "cued") -> Grade:
    workdir = _write_assets(inst, root)
    try:
        sub = agent.solve(inst.view(prompt_style), workdir)
        return fam.grade(inst, sub)
    except Exception as exc:  # crashing on bad data is a failure, not an excuse
        return _error_grade(inst, fam, exc)


def _ordered(fams: list[Family], seeds: list[int], variants,
             shuffle: bool, shuffle_seed: int):
    items = [(f, s, v) for f in fams for s in seeds for v in (variants or f.variants)]
    if not shuffle:
        return items
    rng = random.Random(f"order|{shuffle_seed}")
    rng.shuffle(items)

    def twin(a, b):  # same family + seed => clean/trapped twins
        return a[0].family_id == b[0].family_id and a[1] == b[1]

    for i in range(1, len(items)):       # push apart any adjacent twins
        if twin(items[i], items[i - 1]):
            for j in range(i + 1, len(items)):
                if not twin(items[j], items[i - 1]):
                    items[i], items[j] = items[j], items[i]
                    break
    return items


def run_benchmark(
    agent: Agent,
    *,
    seeds: Iterable[int] = range(5),
    domains: Optional[list[str]] = None,
    families: Optional[list[str]] = None,
    workdir: Optional[str] = None,
    prompt_style: str = "cued",
    shuffle: bool = True,
    shuffle_seed: int = 0,
    progress: bool = False,
) -> list[Grade]:
    """Run ``agent`` over the selected families/seeds; return all grades."""
    seeds = list(seeds)
    fams = registry.select(domains=domains, families=families)
    root = workdir or tempfile.mkdtemp(prefix="shb_")
    items = _ordered(fams, seeds, None, shuffle, shuffle_seed)
    grades: list[Grade] = []
    for k, (fam, seed, variant) in enumerate(items):
        if progress and k % 25 == 0:
            print(f"[shb] {agent.name:18s} {k}/{len(items)}")
        inst = fam.generate(seed, variant)
        grades.append(run_instance(agent, fam, inst, root, prompt_style))
    return grades


def run_family(agent: Agent, fam: Family, seeds: Iterable[int], root: str,
               prompt_style: str = "cued") -> list[Grade]:
    grades = []
    for seed in seeds:
        for variant in fam.variants:
            grades.append(run_instance(agent, fam, fam.generate(seed, variant), root, prompt_style))
    return grades
