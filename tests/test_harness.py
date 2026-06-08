"""Runner mechanics and the LLM adapter's parsing."""
import os

from shb import registry, run_benchmark
from shb.agents import LLMAgent
from shb.types import AgentView, Submission

A_FAMILY = registry.all_families()[0].family_id


class _CrashAgent:
    name = "crash"

    def solve(self, view, workdir):
        raise RuntimeError("boom")


def test_crashing_agent_scores_zero():
    grades = run_benchmark(_CrashAgent(), seeds=[0], families=[A_FAMILY])
    assert grades and all(g.score == 0.0 for g in grades)
    assert any("boom" in str(g.detail.get("error", "")) for g in grades)


def test_assets_written_and_view_is_public(tmp_path):
    captured = {}

    class _Probe:
        name = "probe"

        def solve(self, view, workdir):
            captured["files"] = os.listdir(workdir)
            captured["view"] = view
            return Submission()

    run_benchmark(_Probe(), seeds=[0], families=[A_FAMILY], workdir=str(tmp_path))
    assert captured["files"], "assets should be written to the workdir"
    assert isinstance(captured["view"], AgentView)
    assert not hasattr(captured["view"], "answer")


def test_llm_adapter_parses_messy_json():
    def fake_complete(prompt):
        return ('Sure, here is my analysis. {"answers": {"x": 1}, '
                '"issues_detected": ["unit mismatch"], "abstain": false, "notes": "ok"} Done.')

    view = registry.all_families()[0].generate(0, "clean").view()
    sub = LLMAgent(fake_complete).solve(view, "/tmp")
    assert sub.answers == {"x": 1}
    assert sub.issues_detected == ["unit mismatch"]
    assert sub.abstained is False
