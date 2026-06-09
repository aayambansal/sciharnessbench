"""Isolated, stateless evaluation harness — the official scoring path.

The in-process :func:`shb.run_benchmark` is a transparent development tool: it
runs the agent in the host process, which is convenient but gives a hostile agent
the host's Python (it could ``import shb`` and call an oracle) and a shared
scratch directory. Official scores must not depend on trusting the agent, so they
use this harness instead, which realizes the protocol in ``docs/THREAT_MODEL.md``:

* **One isolated sandbox per task.** Each instance gets its own fresh parent
  directory containing a single ``sandbox/`` dir with only the serialized public
  view (``view.json``) and asset files. ``..`` from the sandbox reveals nothing
  but the sandbox itself; no sibling task is reachable.
* **No state between tasks.** The per-task directory is deleted after grading.
* **Out-of-process, salt-blind.** The agent is an external command run via
  ``subprocess`` with a cleaned environment: ``PYTHONPATH`` stripped and
  ``SHB_SALT``/``SHB_STRICT`` removed, so the agent cannot read the private salt
  or reach the host repo. The non-zero return code, a timeout, a crash, or an
  unparseable submission are each recorded as an *unsafe failure* (score 0),
  distinct from a confident wrong answer.

In the production deployment the agent runs in a container image that does **not**
contain the ``shb`` package and has **no network**, which is the only way to make
``import shb`` and exfiltration impossible; this module is the reference driver of
that protocol and enforces everything achievable from the host side.

The agent command receives the sandbox path as its final argument, reads
``view.json`` + assets, and writes ``submission.json`` of the form
``{"answers": {...}, "issues": [{"kind","evidence"}], "abstain": false,
"confidence": 0.0-1.0}``. See ``scripts/agent_example.py`` for a reference.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import tempfile
from typing import Iterable, Optional

from . import registry
from .runner import _error_grade, _ordered
from .types import Submission, TaskInstance


def _write_sandbox(inst: TaskInstance, taskroot: str) -> str:
    box = os.path.join(taskroot, "sandbox")     # the only child of taskroot: '..' leaks nothing
    os.makedirs(box, exist_ok=True)
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


def _clean_env() -> dict:
    # The agent must not see the host repo or the private salt.
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "SHB_SALT", "SHB_STRICT")}
    env["PYTHONPATH"] = ""
    return env


def run_isolated(agent_cmd, *, seeds: Iterable[int] = range(5),
                 domains=None, families=None, timeout_s: float = 120.0,
                 mem_mb: Optional[int] = None, shuffle_seed: int = 0, progress: bool = False):
    """Run an external ``agent_cmd`` (list[str]) over the suite in isolation.

    Returns a list of Grade. A non-zero exit, timeout, crash, or missing/invalid
    output is an unsafe failure.
    """
    registry.ensure_loaded()
    fams = registry.select(domains=domains, families=families)
    items = _ordered(fams, list(seeds), None, True, shuffle_seed)
    env = _clean_env()
    grades = []
    for k, (fam, seed, variant) in enumerate(items):
        if progress and k % 25 == 0:
            print(f"[shb-isolated] {k}/{len(items)}")
        inst = fam.generate(seed, variant)
        taskroot = tempfile.mkdtemp(prefix="shb_task_")   # unique parent; deleted below
        try:
            box = _write_sandbox(inst, taskroot)
            try:
                proc = subprocess.run(list(agent_cmd) + [box], cwd=box, env=env, timeout=timeout_s,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      preexec_fn=_limit_mem(mem_mb))
                if proc.returncode != 0:
                    grades.append(_error_grade(inst, fam, RuntimeError(f"agent exited {proc.returncode}")))
                else:
                    sub = _parse_submission(box)
                    grades.append(fam.grade(inst, sub) if sub is not None
                                  else _error_grade(inst, fam, RuntimeError("no submission.json")))
            except subprocess.TimeoutExpired:
                grades.append(_error_grade(inst, fam, TimeoutError(f"timeout after {timeout_s}s")))
            except Exception as exc:  # noqa: BLE001
                grades.append(_error_grade(inst, fam, exc))
        finally:
            shutil.rmtree(taskroot, ignore_errors=True)    # no state survives between tasks
    return grades
