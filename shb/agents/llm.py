"""Adapter for testing a real model against SciHarnessBench.

You provide a ``complete(prompt: str) -> str`` callable wrapping any model (an
Anthropic/OpenAI call, a local model, or a full tool-using agent loop). The
adapter renders the task into a prompt, asks for a strict-JSON answer, and parses
the reply into a :class:`Submission`. The JSON schema includes an
``issues_detected`` list and an ``abstain`` flag so the model has a first-class
channel for flagging a flaw it spots — which is exactly what the trapped tasks
test.

This file has no network dependency; it runs only if you pass a working
``complete`` callable. See ``scripts/run_benchmark.py`` for usage.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from ..types import AgentView, Submission

_INSTRUCTIONS = """\
You are a scientific analysis agent. Solve the task using the provided files.
Think like a careful scientist: validate inputs, check units and assumptions,
verify any computation converged, and watch for decoys, confounds, leakage, or
malformed data. If the task cannot be answered correctly as posed, say so.

Reply with ONLY a JSON object of this exact shape:
{
  "answers": { <the requested fields> },
  "issues_detected": [ "<short description of any data/method flaw you found>" ],
  "abstain": <true if the task is not answerable as posed, else false>,
  "notes": "<brief reasoning>"
}
"""


def _render(view: AgentView) -> str:
    parts = [_INSTRUCTIONS, f"\n## Task ({view.domain} / {view.family})\n{view.prompt}"]
    parts.append("\n## Requested answer fields")
    for k, desc in view.answer_fields.items():
        parts.append(f"- {k}: {desc}")
    parts.append("\n## Files")
    for name, content in view.assets.items():
        clip = content if len(content) < 8000 else content[:8000] + "\n...[truncated]"
        parts.append(f"\n### {name}\n```\n{clip}\n```")
    return "\n".join(parts)


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


class LLMAgent:
    def __init__(self, complete: Callable[[str], str], name: str = "llm"):
        self.complete = complete
        self.name = name

    def solve(self, view: AgentView, workdir: str) -> Submission:
        raw = self.complete(_render(view))
        data = _extract_json(raw)
        return Submission(
            answers=data.get("answers", {}) if isinstance(data.get("answers"), dict) else {},
            issues_detected=[str(x) for x in data.get("issues_detected", []) or []],
            abstained=bool(data.get("abstain", False)),
            notes=str(data.get("notes", ""))[:2000],
        )
