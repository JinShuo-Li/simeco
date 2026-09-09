"""Reproducible short-run V5.1/V6 comparison and snapshot-matched ablations."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .reporting import summary
from .simulation import Simulation

ABLATIONS = {
    "receiver_inbox_disabled": {"inbox_enabled": False},
    "sender_transmission_disabled": {"transmission_enabled": False},
    "tokens_permuted": {"token_permutation": [0, 2, 3, 4, 5, 6, 7, 8, 1]},
    "tokens_randomized": {"randomize_received_tokens": True},
    "strengths_randomized": {"randomize_signal_strengths": True},
    "sender_identity_shuffled": {"communication_sender_identity_shuffle": True},
    "social_embeddings_disabled": {"social_embeddings": False},
}


def run_v6_validation(
    steps: int = 2000, seeds=range(3, 8), output: str | Path | None = None
) -> dict:
    split = steps // 2
    rows = []
    for seed in seeds:
        baseline = Simulation(
            seed=seed, learning=True, memory=True, social_memory=True,
            communication=False,
        )
        baseline.run(steps)
        rows.append({"condition": "v5_1_no_communication", **summary(baseline)})

        trained = Simulation(
            seed=seed, learning=True, memory=True, social_memory=True,
            communication=True,
        )
        trained.run(split)
        branches = {"v6_communication": copy.deepcopy(trained)}
        for name, settings in ABLATIONS.items():
            branch = copy.deepcopy(trained)
            for setting, value in settings.items():
                setattr(branch, setting, value)
            branches[name] = branch
        for name, branch in branches.items():
            branch.run(steps - split)
            rows.append({"condition": name, "intervention_step": split, **summary(branch)})
    result = {
        "version": "V6",
        "scope": "short-run evidence only",
        "steps": steps,
        "seeds": list(seeds),
        "snapshot_matched_ablation_step": split,
        "rows": rows,
    }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
