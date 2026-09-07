"""Metrics summaries and experiment log output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .simulation import Simulation


def summary(simulation: Simulation) -> dict:
    herbivores = simulation.species("herbivore")
    predators = simulation.species("predator")
    steps = max(1, simulation.step_count)
    prey_actions = [sum(a.action_counts[i] for a in herbivores) for i in range(5)]
    predator_actions = [sum(a.action_counts[i] for a in predators) for i in range(5)]
    return {
        "seed": simulation.seed,
        "learning": simulation.learning,
        "step": simulation.step_count,
        "herbivores": len(herbivores),
        "predators": len(predators),
        "mean_plant_biomass": round(sum(map(sum, simulation.resources)) / (simulation.config.width * simulation.config.height), 4),
        "births_herbivore": simulation.metrics.births_herbivore,
        "births_predator": simulation.metrics.births_predator,
        "deaths_predation": simulation.metrics.deaths_predation,
        "deaths_starvation": simulation.metrics.deaths_starvation,
        "deaths_age": simulation.metrics.deaths_age,
        "hunt_success": round(simulation.metrics.hunts / max(1, simulation.metrics.hunt_attempts), 5),
        "herbivore_reward_per_step": round(simulation.metrics.reward_herbivore / steps, 4),
        "predator_reward_per_step": round(simulation.metrics.reward_predator / steps, 4),
        "learning_updates": simulation.metrics.learning_updates,
        "max_generation": max((a.generation for a in simulation.organisms.values()), default=0),
        "prey_idle_fraction": round(prey_actions[0] / max(1, sum(prey_actions)), 4),
        "predator_idle_fraction": round(predator_actions[0] / max(1, sum(predator_actions)), 4),
    }


def write_history(simulation: Simulation, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = simulation.metrics.history
    if not rows:
        return destination
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return destination


def print_summary(simulation: Simulation) -> None:
    print(json.dumps(summary(simulation), indent=2, sort_keys=True))
