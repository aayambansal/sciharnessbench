"""Runner mechanics, opaque workdirs, and the LLM adapter parsing."""
import os

from shb import registry, run_benchmark
from shb.agents import LLMAgent
from shb.types import AgentView, Submission

A_FAMILY = registry.all_families()[0].family_id


class _CrashAgent:
    name = "crash"

    def solve(self, view, workdir):
        raise RuntimeError("boom")


def test_crash_scores_zero_and_is_not_confident_wrong():
    grades = run_benchmark(_CrashAgent(), seeds=[0], families=[A_FAMILY])
    assert grades and all(g.score == 0.0 for g in grades)
    assert all(not g.confident_wrong for g in grades)            # a crash is unsafe failure, not fake science
    assert any("boom" in str(g.detail.get("error", "")) for g in grades)


def test_workdir_is_opaque(tmp_path):
    seen = {}

    class _Probe:
        name = "probe"

        def solve(self, view, workdir):
            seen["dir"] = os.path.basename(workdir)
            seen["view"] = view
            return Submission()

    run_benchmark(_Probe(), seeds=[0], families=[A_FAMILY], workdir=str(tmp_path))
    low = seen["dir"].lower()
    assert "clean" not in low and "trapped" not in low and "seed" not in low
    assert isinstance(seen["view"], AgentView) and not hasattr(seen["view"], "variant")


def test_llm_adapter_parses_structured_issues():
    def fake_complete(prompt):
        return ('Analysis. {"answers": {"x": 1}, "issues": [{"kind": "unit_mismatch", '
                '"evidence": {"unit": "kcal"}}], "abstain": false, "confidence": 0.7, "notes": "ok"}')

    view = registry.all_families()[0].generate(0, "clean").view()
    sub = LLMAgent(fake_complete).solve(view, "/tmp")
    assert sub.answers == {"x": 1}
    assert sub.issues and sub.issues[0]["kind"] == "unit_mismatch"
    assert sub.confidence == 0.7 and sub.abstained is False
