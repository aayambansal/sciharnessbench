"""The benchmark's core invariant: it separates real science from fake science.

A careful agent (validates inputs) must pass both clean and trapped tasks; a
naive agent (trusts inputs) must pass clean tasks but fall for the traps. If
this fails, the benchmark is not measuring what it claims.
"""
import pytest

from shb import aggregate, run_benchmark
from shb.agents import ReferenceAgent

SEEDS = range(10)


@pytest.fixture(scope="module")
def cards():
    naive = aggregate("naive", run_benchmark(ReferenceAgent("naive"), seeds=SEEDS))
    careful = aggregate("careful", run_benchmark(ReferenceAgent("careful"), seeds=SEEDS))
    return naive, careful


def test_careful_does_real_science(cards):
    _, careful = cards
    assert careful.overall["competence"] == 1.0, "careful must solve every clean task"
    assert careful.overall["robustness"] == 1.0, "careful must survive every trap"
    assert careful.overall["fake_science_gap"] == 0.0


def test_naive_is_competent_but_fooled(cards):
    naive, _ = cards
    assert naive.overall["competence"] == 1.0, "naive must solve clean tasks (tasks are doable)"
    assert naive.overall["robustness"] < 0.05, "naive must fall for nearly every trap"
    assert naive.overall["fake_science_gap"] > 0.9


def test_every_family_discriminates(cards):
    naive, careful = cards
    for fid, b in careful.by_family.items():
        assert b["competence"] == 1.0, f"careful not competent on {fid}"
        assert b["robustness"] == 1.0, f"careful not robust on {fid}"
    for fid, b in naive.by_family.items():
        assert b["robustness"] < 0.4, f"trap in {fid} barely bites the naive agent"


def test_all_trap_types_exercised(cards):
    _, careful = cards
    from shb.taxonomy import TrapType
    covered = set(careful.by_trap)
    assert covered == {t.value for t in TrapType}, "every taxonomy trap type must appear"
