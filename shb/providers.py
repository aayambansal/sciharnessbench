"""Provider adapters: turn a model id into a ``complete(prompt) -> str`` callable.

Each adapter reads its key from the environment (``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, ``GOOGLE_API_KEY``) so the same code runs locally and in a
Modal container with injected secrets. Wrap the callable in
:class:`shb.agents.LLMAgent` to get an Agent. A model is named ``provider:id``
(e.g. ``openai:gpt-5.5``, ``anthropic:claude-opus-4-8``, ``google:gemini-3-pro-preview``).

All calls are defensive: any provider error returns ``""`` so the agent yields an
empty submission (a graded miss), never a crash that would abort a run.
"""
from __future__ import annotations

import os
from typing import Callable

from .agents import LLMAgent

_OPENAI_REASONING = ("gpt-5", "o1", "o3", "o4")   # families that take a reasoning param


def openai_complete(model: str, effort: str = "low") -> Callable[[str], str]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=300, max_retries=3)
    reasoning = any(model.startswith(p) for p in _OPENAI_REASONING)

    def complete(prompt: str) -> str:
        try:
            kw = {"model": model, "input": prompt}
            if reasoning:
                kw["reasoning"] = {"effort": effort}
            return client.responses.create(**kw).output_text or ""
        except Exception as exc:  # noqa: BLE001
            return ""
    return complete


def anthropic_complete(model: str, max_tokens: int = 4096) -> Callable[[str], str]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=300, max_retries=3)

    def complete(prompt: str) -> str:
        try:
            msg = client.messages.create(model=model, max_tokens=max_tokens,
                                         messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        except Exception:  # noqa: BLE001
            return ""
    return complete


def google_complete(model: str) -> Callable[[str], str]:
    from google import genai
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def complete(prompt: str) -> str:
        try:
            return client.models.generate_content(model=model, contents=prompt).text or ""
        except Exception:  # noqa: BLE001
            return ""
    return complete


_FACTORY = {"openai": openai_complete, "anthropic": anthropic_complete, "google": google_complete}


def build_agent(spec: str, **kw) -> LLMAgent:
    """``spec`` is ``provider:model_id`` -> an :class:`LLMAgent`."""
    provider, model = spec.split(":", 1)
    return LLMAgent(_FACTORY[provider](model, **({"effort": kw["effort"]} if provider == "openai" and "effort" in kw else {})),
                    name=spec)
