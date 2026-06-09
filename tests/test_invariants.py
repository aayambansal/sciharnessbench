"""Generation invariants, fuzzed over many seeds and multiple salts.

Rather than trust a single reference run, we assert the structural guarantees of
every family across seeds and salts: the careful oracle passes, the naive oracle
fails only the traps, twins share a base, and the trap actually bites. Grading
goes through ``run_instance`` so a naive crash on corrupt input is handled the way
the harness handles it (an unsafe failure that fails the trap), not as a test error.
"""
import os
import tempfile

import pytest

from shb import registry, run_instance
from shb.agents import ReferenceAgent

SEEDS = range(6)
SALTS = ["", "probe7"]
FAMS = registry.all_families()
IDS = [f.family_id for f in FAMS]
CAREFUL, NAIVE = ReferenceAgent("careful"), ReferenceAgent("naive")
ROOT = tempfile.mkdtemp(prefix="shb_inv_")


@pytest.fixture(autouse=True)
def _restore_salt():
    old = os.environ.get("SHB_SALT")
    yield
    if old is None:
        os.environ.pop("SHB_SALT", None)
    else:
        os.environ["SHB_SALT"] = old


@pytest.mark.parametrize("fam", FAMS, ids=IDS)
def test_family_invariants_across_seeds_and_salts(fam):
    for salt in SALTS:
        os.environ["SHB_SALT"] = salt
        for seed in SEEDS:
            clean, trapped = fam.generate(seed, "clean"), fam.generate(seed, "trapped")
            tag = f"{fam.family_id}/salt={salt!r}/seed={seed}"

            assert run_instance(CAREFUL, fam, clean, ROOT).conclusion_correct, f"careful failed clean {tag}"
            assert run_instance(CAREFUL, fam, trapped, ROOT).conclusion_correct, f"careful failed trapped {tag}"
            assert run_instance(NAIVE, fam, clean, ROOT).conclusion_correct, f"naive failed clean {tag}"
            # The trap bites: the input-trusting method does not pass the trapped twin
            # (whether by a confident wrong answer or an unsafe crash).
            assert not run_instance(NAIVE, fam, trapped, ROOT).conclusion_correct, f"trap did not bite {tag}"

            if fam.paired:
                assert clean.grading.get("base_signature") == trapped.grading.get("base_signature") \
                    and clean.grading.get("base_signature") is not None, f"twin base mismatch {tag}"
