"""Metrics summaries and experiment log output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .actions import Effort, Interaction, Locomotion
from .perception import OBSERVATION_SIZE, channel_index
from .simulation import Simulation


def summary(simulation: Simulation) -> dict:
    herbivores = simulation.species("herbivore")
    predators = simulation.species("predator")
    steps = max(1, simulation.step_count)
    prey_actions = simulation.metrics.actions_herbivore
    predator_actions = simulation.metrics.actions_predator
    def probe(energy: float = 0.5, food_here: float = 0.0) -> list[float]:
        result = [1.0, energy, 0.25, 1.0 - energy, food_here, 0.0]
        return result + [0.0] * (OBSERVATION_SIZE - len(result))

    def probabilities(animal, observation):
        return animal.arbiter.probabilities(
            animal.instinct.preferences(observation),
            animal.adaptive_policy.preferences(observation),
            simulation.learning,
        )

    hungry_food = probe(energy=0.2)
    hungry_food[channel_index("plants", 1, 0)] = 1.0
    satiated_food = probe(energy=0.9)
    satiated_food[channel_index("plants", 1, 0)] = 1.0
    hungry_on_food = probe(energy=0.2, food_here=1.0)
    satiated_on_food = probe(energy=0.9, food_here=1.0)
    danger_ahead = probe(energy=0.5)
    danger_ahead[channel_index("predators", 1, 0)] = 1.0
    prey_ahead = probe(energy=0.35)
    prey_ahead[channel_index("herbivores", 1, 0)] = 1.0
    prey_here = probe(energy=0.35)
    prey_here[channel_index("herbivores", 0, 0)] = 1.0
    satiated = probe(energy=0.9)
    hungry = probe(energy=0.2)

    food_approach = sum(probabilities(a, hungry_food)[Locomotion.FORWARD] for a in herbivores) / max(1, len(herbivores))
    satiated_food_approach = sum(
        probabilities(a, satiated_food)[Locomotion.FORWARD] for a in herbivores
    ) / max(1, len(herbivores))
    hungry_feed = sum(
        probabilities(a, hungry_on_food)[7 + Interaction.FEED] for a in herbivores
    ) / max(1, len(herbivores))
    satiated_feed = sum(
        probabilities(a, satiated_on_food)[7 + Interaction.FEED] for a in herbivores
    ) / max(1, len(herbivores))
    flee_turn = sum(
        probabilities(a, danger_ahead)[Locomotion.TURN_LEFT]
        + probabilities(a, danger_ahead)[Locomotion.TURN_RIGHT]
        for a in herbivores
    ) / max(1, len(herbivores))
    flee_sprint = sum(probabilities(a, danger_ahead)[4 + Effort.SPRINT] for a in herbivores) / max(1, len(herbivores))
    pursuit = sum(probabilities(a, prey_ahead)[Locomotion.FORWARD] for a in predators) / max(1, len(predators))
    attack = sum(probabilities(a, prey_here)[7 + Interaction.ATTACK] for a in predators) / max(1, len(predators))
    conserve = sum(probabilities(a, satiated)[4 + Effort.LOW] for a in predators) / max(1, len(predators))
    hungry_search = sum(
        probabilities(a, hungry)[Locomotion.FORWARD] for a in predators
    ) / max(1, len(predators))
    satiated_search = sum(
        probabilities(a, satiated)[Locomotion.FORWARD] for a in predators
    ) / max(1, len(predators))
    all_efforts = sum(simulation.metrics.efforts_herbivore) + sum(simulation.metrics.efforts_predator)
    return {
        "seed": simulation.seed,
        "learning": simulation.learning,
        "controller_mode": simulation.controller_mode,
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
        "prey_escape_rate": round(simulation.metrics.prey_escapes / max(1, simulation.metrics.hunt_attempts), 5),
        "herbivore_reward_per_step": round(simulation.metrics.reward_herbivore / steps, 4),
        "predator_reward_per_step": round(simulation.metrics.reward_predator / steps, 4),
        "learning_updates": simulation.metrics.learning_updates,
        "max_generation": max((a.generation for a in simulation.organisms.values()), default=0),
        "prey_idle_fraction": round(prey_actions[0] / max(1, sum(prey_actions)), 4),
        "predator_idle_fraction": round(predator_actions[0] / max(1, sum(predator_actions)), 4),
        "plant_food_per_herbivore_action": round(
            simulation.metrics.plants_eaten / max(1, sum(prey_actions)), 4
        ),
        "hunts_per_1000_predator_actions": round(
            1000.0 * simulation.metrics.hunts / max(1, sum(predator_actions)), 4
        ),
        "food_energy_per_herbivore_energy_spent": round(
            simulation.metrics.energy_gained_herbivore
            / max(0.001, simulation.metrics.energy_spent_herbivore), 4
        ),
        "hunt_energy_per_predator_energy_spent": round(
            simulation.metrics.energy_gained_predator
            / max(0.001, simulation.metrics.energy_spent_predator), 4
        ),
        "herbivore_sprint_fraction": round(
            simulation.metrics.efforts_herbivore[Effort.SPRINT]
            / max(1, sum(simulation.metrics.efforts_herbivore)), 4
        ),
        "predator_sprint_fraction": round(
            simulation.metrics.efforts_predator[Effort.SPRINT]
            / max(1, sum(simulation.metrics.efforts_predator)), 4
        ),
        "unnecessary_sprint_fraction": round(
            simulation.metrics.unnecessary_sprints / max(1, all_efforts), 4
        ),
        "herbivore_births_per_1000_intents": round(
            1000.0 * simulation.metrics.births_herbivore
            / max(1, simulation.metrics.reproduction_intents_herbivore), 4
        ),
        "predator_births_per_1000_intents": round(
            1000.0 * simulation.metrics.births_predator
            / max(1, simulation.metrics.reproduction_intents_predator), 4
        ),
        "hungry_food_approach": round(food_approach, 4),
        "hunger_food_approach_delta": round(food_approach - satiated_food_approach, 4),
        "hunger_feed_delta": round(hungry_feed - satiated_feed, 4),
        "prey_flee_turn": round(flee_turn, 4),
        "prey_flee_sprint": round(flee_sprint, 4),
        "predator_pursuit": round(pursuit, 4),
        "predator_attack": round(attack, 4),
        "predator_energy_conservation": round(conserve, 4),
        "predator_hunger_search_delta": round(hungry_search - satiated_search, 4),
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
