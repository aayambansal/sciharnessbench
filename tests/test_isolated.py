"""The isolated official evaluator: isolation properties and failure handling."""
import json
import os
import sys

from shb import registry
from shb.isolated import _clean_env, _write_sandbox, run_isolated

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_clean_env_strips_salt_and_path(monkeypatch):
    monkeypatch.setenv("SHB_SALT", "secret-official-salt")
    monkeypatch.setenv("PYTHONPATH", "/path/to/repo")
    env = _clean_env()
    assert "SHB_SALT" not in env, "the agent must not be able to read the private salt"
    assert "SHB_STRICT" not in env
    assert env["PYTHONPATH"] == "", "the host repo must not be on the agent's path"


def test_sandbox_hides_siblings_and_ground_truth(tmp_path):
    inst = registry.get("chem.reaction_energy").generate(0, "trapped")
    box = _write_sandbox(inst, str(tmp_path))
    # '..' from the sandbox reveals only the sandbox itself — no sibling task.
    assert sorted(os.listdir(str(tmp_path))) == ["sandbox"]
    v = json.load(open(os.path.join(box, "view.json")))
    for hidden in ("answer", "grading", "variant", "seed", "trap_type"):
        assert hidden not in v
    assert v["task_id"].startswith("task_") and "trapped" not in v["task_id"]


def test_nonzero_exit_is_unsafe_failure():
    grades = run_isolated([sys.executable, "-c", "import sys; sys.exit(3)"],
                          seeds=range(1), families=["chem.reaction_energy"], timeout_s=30)
    assert grades[0].score == 0.0 and not grades[0].confident_wrong
    assert grades[0].detail.get("error")


def test_example_agent_runs_end_to_end():
    grades = run_isolated([sys.executable, os.path.join(REPO, "scripts", "agent_example.py")],
                          seeds=range(1), families=["chem.reaction_energy", "stats.simpson"], timeout_s=60)
    assert len(grades) == 4 and all(g.score in (0.0, 1.0) for g in grades)
