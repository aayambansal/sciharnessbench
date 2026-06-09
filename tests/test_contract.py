"""Every registered family must honor the v2 task contract."""
import pytest

from shb import registry
from shb.types import allowed_issue_kinds

FAMS = registry.all_families()
IDS = [f.family_id for f in FAMS]
KINDS = set(allowed_issue_kinds())


@pytest.mark.parametrize("fam", FAMS, ids=IDS)
def test_clean_and_trapped_shapes(fam):
    clean, trapped = fam.generate(0, "clean"), fam.generate(0, "trapped")
    assert clean.variant == "clean" and clean.trap_type is None
    assert trapped.variant == "trapped" and trapped.trap_type is not None
    assert clean.assets and trapped.assets
    assert clean.answer_fields == trapped.answer_fields
    assert trapped.trap_type in fam.trap_types
    assert fam.flaw_kind in KINDS


@pytest.mark.parametrize("fam", FAMS, ids=IDS)
def test_view_is_opaque_and_hides_everything(fam):
    inst = fam.generate(7, "trapped")
    view = inst.view()
    low = view.task_id.lower()
    assert view.task_id.startswith("task_")
    assert "clean" not in low and "trapped" not in low and "seed" not in low
    assert view.task_id != inst.task_id            # public != internal id
    for hidden in ("variant", "seed", "answer", "grading", "trap_type", "trap_note"):
        assert not hasattr(view, hidden), f"view leaked {hidden}"
    assert set(view.allowed_issue_kinds) == KINDS


@pytest.mark.parametrize("fam", FAMS, ids=IDS)
def test_deterministic(fam):
    for v in ("clean", "trapped"):
        a, b = fam.generate(3, v), fam.generate(3, v)
        assert a.assets == b.assets and a.prompt == b.prompt and a.answer == b.answer


@pytest.mark.parametrize("fam", FAMS, ids=IDS)
def test_distinct_seeds_distinct_problems(fam):
    assert fam.generate(0, "clean").assets != fam.generate(1, "clean").assets


@pytest.mark.parametrize("fam", [f for f in FAMS if f.paired], ids=[f.family_id for f in FAMS if f.paired])
def test_paired_twins_share_base_signature(fam):
    # Paired families must be true counterfactual twins: identical base, only the trap differs.
    for seed in range(4):
        c = fam.generate(seed, "clean").grading.get("base_signature")
        t = fam.generate(seed, "trapped").grading.get("base_signature")
        assert c is not None and c == t, f"{fam.family_id} twins differ in base at seed {seed}"
