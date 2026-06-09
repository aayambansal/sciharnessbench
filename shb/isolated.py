"""Isolated, stateless evaluation harness — the official scoring path.

The in-process :func:`shb.run_benchmark` is a transparent development tool: it
runs the agent in the host process, which is convenient but gives a hostile agent
the host's Python (it could ``import shb`` and call an oracle) and a shared
scratch directory. Official scores must not depend on trusting the agent, so they
use this harness instead, which realizes the protocol in ``docs/THREAT_MODEL.md``:

* **One sandbox per task.** Each instance gets a fresh, unique temp directory with
  only its serialized public view (``view.json``) and asset files. No sibling task
  is reachable; no state survives between tasks.
* **Out-of-process.** The agent is an external command run via ``subprocess`` with
  a cleaned environment (``PYTHONPATH`` stripped) and the sandbox as its cwd. In
  the production deployment the agent runs in a container that does not contain
  the ``shb`` package at all, so the generators, graders, and reference solvers
  are unreachable.
* **Bounded.** A per-task wall-clock timeout (and optional memory cap on POSIX)
  applies; a timeout, crash, or unparseable submission is recorded as an *unsafe
  failure* (score 0) — distinct from a confident wrong answer.

The agent command receives the sandbox path as its final argument, reads
``view.json`` + assets, and writes ``submission.json`` of the form
``{"answers": {...}, "issues": [{"kind","evidence"}], "abstain": false,
"confidence": 0.0-1.0}``. See ``scripts/agent_example.py`` for a reference.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import tempfile
from typing import Iterable, Optional

from . import registry
from .runner import _error_grade, _ordered
from .types import Submission, TaskInstance


def _write_sandbox(inst: TaskInstance, root: str) -> str:
    box = tempfile.mkdtemp(prefix="task_", dir=root)
    view = inst.view()
    for name, content in view.assets.items():
        with open(os.path.join(box, name), "w") as fh:
            fh.write(content)
    json.dump({"task_id": view.task_id, "domain": view.domain, "family": view.family,
               "prompt": view.prompt, "answer_fields": view.answer_fields,
               "allowed_issue_kinds": view.allowed_issue_kinds,
               "assets": list(view.assets)}, open(os.path.join(box, "view.json"), "w"))
    return box


def _limit_mem(mb: Optional[int]):
    if mb is None:
        return None

    def _set():
        try:
            soft = mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
        except (ValueError, OSError):
            pass
    return _set


def _parse_submission(box: str) -> Optional[Submission]:
    path = os.path.join(box, "submission.json")
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path))
    except (ValueError, OSError):
        return None
    issues = [i for i in (d.get("issues") or []) if isinstance(i, dict) and "kind" in i]
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence", 1.0))))
    except (TypeError, ValueError):
        conf = 1.0
    return Submission(answers=d.get("answers", {}) if isinstance(d.get("answers"), dict) else {},
                      issues=issues, abstained=bool(d.get("abstain", False)),
                      confidence=conf, notes=str(d.get("notes", ""))[:2000])


def run_isolated(agent_cmd, *, seeds: Iterable[int] = range(5),
                 domains=None, families=None, timeout_s: float = 120.0,
                 mem_mb: Optional[int] = None, shuffle_seed: int = 0, progress: bool = False):
    """Run an external ``agent_cmd`` (list[str]) over the suite in isolation.

    Returns a list of Grade. A timeout/crash/missing-output is an unsafe failure.
    """
    registry.ensure_loaded()
    fams = registry.select(domains=domains, families=families)
    items = _ordered(fams, list(seeds), None, True, shuffle_seed)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = ""           # the host repo is not on the agent's path
    root = tempfile.mkdtemp(prefix="shb_isolated_")
    grades = []
    for k, (fam, seed, variant) in enumerate(items):
        if progress and k % 25 == 0:
            print(f"[shb-isolated] {k}/{len(items)}")
        inst = fam.generate(seed, variant)
        box = _write_sandbox(inst, root)
        try:
            subprocess.run(list(agent_cmd) + [box], cwd=box, env=env, timeout=timeout_s,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           preexec_fn=_limit_mem(mem_mb))
            sub = _parse_submission(box)
            grades.append(fam.grade(inst, sub) if sub is not None
                          else _error_grade(inst, fam, RuntimeError("no submission.json")))
        except subprocess.TimeoutExpired:
            grades.append(_error_grade(inst, fam, TimeoutError(f"timeout after {timeout_s}s")))
        except Exception as exc:  # noqa: BLE001
            grades.append(_error_grade(inst, fam, exc))
    return grades
