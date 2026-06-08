"""Run an agent over task families and collect grades.

The runner is deliberately dumb: for each (family, seed, variant) it generates
an instance, writes its assets into a scratch dir, hands the agent the public
view, and grades the result. An agent that raises (e.g. crashing on a corrupt
input) is scored as a failed conclusion — crashing on bad data is itself a
fake-science failure, not an excuse.
"""
from __future__ import annotations

import os
import tempfile
import traceback
from typing import Iterable, Optional, Sequence

from . import registry
from .types import Agent, Family, Grade, TaskInstance


def _write_assets(inst: TaskInstance, root: str) -> str:
    d = os.path.join(root, inst.task_id.replace("/", "__"))
    os.makedirs(d, exist_ok=True)
    for name, content in inst.assets.items():
        with open(os.path.join(d, name), "w") as fh:
            fh.write(content)
    return d


def run_instance(agent: Agent, inst: TaskInstance, root: str) -> Grade:
    fam = registry.get(inst.family)
    workdir = _write_assets(inst, root)
    try:
        submission = agent.solve(inst.view(), workdir)
        grade = fam.grade(inst, submission)
    except Exception as exc:  # an agent that crashes failed the task
        grade = Grade(
            task_id=inst.task_id, family=inst.family, domain=inst.domain,
            variant=inst.variant,
            trap_type=inst.trap_type.value if inst.trap_type else None,
            conclusion_correct=False,
            trap_detected=False if inst.is_trapped else None,
            score=0.0,
            detail={"error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().splitlines()[-3:]},
        )
    return grade


def run_family(agent: Agent, fam: Family, seeds: Iterable[int],
               root: str, variants: Optional[Sequence[str]] = None) -> list[Grade]:
    variants = variants or fam.variants
    grades: list[Grade] = []
    for seed in seeds:
        for variant in variants:
            inst = fam.generate(seed, variant)
            grades.append(run_instance(agent, inst, root))
    return grades


def run_benchmark(
    agent: Agent,
    *,
    seeds: Iterable[int] = range(5),
    domains: Optional[list[str]] = None,
    families: Optional[list[str]] = None,
    workdir: Optional[str] = None,
    progress: bool = False,
) -> list[Grade]:
    """Run ``agent`` over the selected families and seeds; return all grades."""
    seeds = list(seeds)
    fams = registry.select(domains=domains, families=families)
    root = workdir or tempfile.mkdtemp(prefix="shb_")
    grades: list[Grade] = []
    for fam in fams:
        if progress:
            print(f"[shb] {agent.name:16s} {fam.family_id} x{len(seeds)} seeds")
        grades.extend(run_family(agent, fam, seeds, root))
    return grades
