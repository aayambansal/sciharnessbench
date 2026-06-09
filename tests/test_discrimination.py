"""The core invariant: the benchmark separates real science from fake science."""
import pytest

from shb import aggregate, registry, run_benchmark
from shb.agents import ReferenceAgent
from shb.taxonomy import TrapType

SEEDS = range(10)


@pytest.fixture(scope="module")
def cards():
    naive = aggregate("naive", run_benchmark(ReferenceAgent("naive"), seeds=SEEDS))
    careful = aggregate("careful", run_benchmark(ReferenceAgent("careful"), seeds=SEEDS))
    return naive, careful


def test_careful_does_real_science(cards):
    _, c = cards
    h = c.headline
    assert h["competence"] == 1.0
    assert h["robustness"] == 1.0
    assert h["fake_science_gap"] == 0.0
    assert h["false_alarm_rate"] == 0.0          # never cries wolf on clean
    assert h["trap_detection_rate"] == 1.0


def test_naive_is_competent_but_fooled(cards):
    n, _ = cards
    h = n.headline
    assert h["competence"] == 1.0                 # the science is doable
    assert h["robustness"] < 0.05                 # falls for the traps
    assert h["fake_science_gap"] > 0.9


def test_every_paired_family_discriminates(cards):
    naive, careful = cards
    for fid, b in careful.by_family.items():
        assert b["competence"] == 1.0 and b["robustness"] == 1.0, fid
    for fid, b in naive.by_family.items():
        assert b["robustness"] < 0.4, fid


def test_nonpaired_scenarios_also_discriminate(cards):
    naive, careful = cards
    for fid, b in careful.scenarios.get("families", {}).items():
        assert b["robustness"] == 1.0, fid
    for fid, b in naive.scenarios.get("families", {}).items():
        assert b["robustness"] < 0.4, fid


def test_all_trap_types_are_covered():
    covered = {t.value for f in registry.all_families() for t in f.trap_types}
    assert covered == {t.value for t in TrapType}, "every taxonomy trap type must be exercised"
