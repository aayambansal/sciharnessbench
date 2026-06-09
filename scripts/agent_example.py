#!/usr/bin/env python3
"""Reference external agent for the isolated harness (shb/isolated.py).

Run as:  python agent_example.py <sandbox_dir>

It reads ``<sandbox>/view.json`` and the asset files, decides, and writes
``<sandbox>/submission.json``. It does NOT import ``shb`` — in the official
deployment the benchmark package is not present in the agent's environment. To
test a real model, replace :func:`decide` with a call to your model/agent loop
that returns the same four values.
"""
import json
import os
import sys


def decide(view: dict, assets: dict):
    """Return (answers, issues, abstain, confidence).

    ``view`` has keys: task_id, domain, family, prompt, answer_fields,
    allowed_issue_kinds. ``assets`` maps filename -> text. ``issues`` is a list of
    ``{"kind": <one of view['allowed_issue_kinds']>, "evidence": <checkable detail>}``.

    This template makes no claim (it is a safe, non-committal baseline). Plug your
    model here: read the prompt and assets, do the analysis, validate the inputs,
    and report any flaw you can substantiate with evidence.
    """
    return {}, [], False, 0.0


def main():
    box = sys.argv[1]
    view = json.load(open(os.path.join(box, "view.json")))
    assets = {n: open(os.path.join(box, n)).read() for n in view["assets"]}
    answers, issues, abstain, confidence = decide(view, assets)
    json.dump({"answers": answers, "issues": issues, "abstain": abstain,
               "confidence": confidence}, open(os.path.join(box, "submission.json"), "w"))


if __name__ == "__main__":
    main()
