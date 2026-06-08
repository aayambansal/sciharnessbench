"""Every registered family must honor the task contract."""
import pytest

from shb import registry

FAMS = registry.all_families()
IDS = [f.family_id for f in FAMS]


@pytest.mark.parametrize("fam", FAMS, ids=IDS)
def test_clean_and_trapped_shapes(fam):
    clean = fam.generate(0, "clean")
    trapped = fam.generate(0, "trapped")
    assert clean.variant == "clean" and clean.trap_type is None
    assert trapped.variant == "trapped" and trapped.trap_type is not None
    assert clean.assets and trapped.assets, "both variants must ship assets"
    assert clean.answer_fields == trapped.answer_fields, "twins ask the same question"
    assert trapped.trap_type in fam.trap_types


@pytest.mark.parametrize("fam", FAMS, ids=IDS)
def test_view_hides_ground_truth(fam):
    view = fam.generate(0, "trapped").view()
    # The agent must never see the answer, the grading payload, or the trap label.
    for hidden in ("answer", "grading", "trap_type", "trap_note"):
        assert not hasattr(view, hidden), f"view leaked {hidden}"


@pytest.mark.parametrize("fam", FAMS, ids=IDS)
def test_deterministic(fam):
    for variant in ("clean", "trapped"):
        a = fam.generate(3, variant)
        b = fam.generate(3, variant)
        assert a.assets == b.assets and a.prompt == b.prompt and a.answer == b.answer


@pytest.mark.parametrize("fam", FAMS, ids=IDS)
def test_distinct_seeds_give_distinct_problems(fam):
    # Guards the seeding pitfall: prefix-sharing seeds must not collapse to one
    # problem. Different seeds must produce different instances.
    a = fam.generate(0, "clean")
    b = fam.generate(1, "clean")
    assert a.assets != b.assets, f"{fam.family_id} produces identical assets across seeds"
