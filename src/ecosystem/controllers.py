"""Innate behavior and arbitration between instinct and adaptive preferences."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .actions import HEAD_SIZES, Effort, EmbodiedAction, Interaction, Locomotion, Reproduction
from .perception import channel_index


@dataclass(slots=True)
class InstinctController:
    """Species-specific survival behavior available unchanged from birth."""

    species: str
    primary_strength: float = 4.0
    reproduction_energy_fraction: float = 0.7
    maturity_fraction: float = 0.02
    decisions: int = 0

    def preferences(self, observation: list[float]) -> list[float]:
        energy = observation[1]
        hunger = observation[3]
        plant_here = observation[4]
        locomotion = [0.0] * HEAD_SIZES[0]
        effort = [0.0] * HEAD_SIZES[1]
        interaction = [0.0] * HEAD_SIZES[2]
        reproduction = [0.0] * HEAD_SIZES[3]

        def sensory_grid(channel: str) -> list[tuple[float, int, int]]:
            return [
                (observation[channel_index(channel, forward, lateral)], forward, lateral)
                for forward in (-1, 0, 1)
                for lateral in (-1, 0, 1)
            ]

        def approach(forward: int, lateral: int) -> int:
            if forward > 0 and lateral == 0:
                return Locomotion.FORWARD
            if lateral < 0:
                return Locomotion.TURN_LEFT
            if lateral > 0:
                return Locomotion.TURN_RIGHT
            if forward < 0:
                return Locomotion.TURN_LEFT
            return Locomotion.HOLD

        def avoid(forward: int, lateral: int, grid: list[tuple[float, int, int]]) -> int:
            if forward < 0 and lateral == 0:
                return Locomotion.FORWARD
            if lateral < 0:
                return Locomotion.TURN_RIGHT
            if lateral > 0:
                return Locomotion.TURN_LEFT
            left_danger = sum(value for value, _, side in grid if side < 0)
            right_danger = sum(value for value, _, side in grid if side > 0)
            return Locomotion.TURN_LEFT if left_danger <= right_danger else Locomotion.TURN_RIGHT

        if self.species == "herbivore":
            plants = sensory_grid("plants")
            predators = sensory_grid("predators")
            best_food, food_forward, food_lateral = max(plants)
            danger, danger_forward, danger_lateral = max(predators)
            locomotion[Locomotion.HOLD] += plant_here * (1.5 + 2.5 * hunger)
            locomotion[approach(food_forward, food_lateral)] += 2.0 * best_food * hunger
            if danger > 0.0:
                locomotion[avoid(danger_forward, danger_lateral, predators)] += (
                    self.primary_strength * danger
                )
                locomotion[Locomotion.HOLD] -= 2.0 * danger
                effort[Effort.SPRINT] += 4.0 * danger
            effort[Effort.LOW] += 1.5 * (1.0 - danger) * (1.0 - hunger)
            effort[Effort.CRUISE] += 1.0 + hunger
            interaction[Interaction.FEED] += 5.0 * plant_here * (0.4 + hunger)
            interaction[Interaction.NONE] += 1.5 * (1.0 - plant_here)
            interaction[Interaction.ATTACK] -= 4.0
        elif self.species == "predator":
            prey = sensory_grid("herbivores")
            target, target_forward, target_lateral = max(prey)
            if target > 0.0:
                locomotion[approach(target_forward, target_lateral)] += self.primary_strength * target
                effort[Effort.CRUISE] += 1.5 * target
                effort[Effort.SPRINT] += 2.2 * target * hunger
            else:
                locomotion[Locomotion.HOLD] += 1.5 * energy
                locomotion[Locomotion.FORWARD] += 0.7 * hunger
                effort[Effort.LOW] += 2.0 * energy
                effort[Effort.CRUISE] += 0.8 * hunger
            prey_here = observation[channel_index("herbivores", 0, 0)]
            interaction[Interaction.ATTACK] += 6.0 * prey_here * (0.5 + hunger)
            interaction[Interaction.NONE] += 2.0 * (1.0 - prey_here)
            interaction[Interaction.FEED] -= 4.0
        else:
            raise ValueError(f"unknown species: {self.species}")

        eligible = energy >= self.reproduction_energy_fraction and observation[2] >= self.maturity_fraction
        reproduction[Reproduction.INTEND if eligible else Reproduction.DEFER] += 4.0
        return locomotion + effort + interaction + reproduction

    def reproduction_tendency(
        self, observation: list[float], maturity_fraction: float, threshold_fraction: float
    ) -> float:
        """Innate gate for energy- and maturity-dependent reproduction."""
        if observation[1] < threshold_fraction or observation[2] < maturity_fraction:
            return 0.0
        return min(1.0, (observation[1] - threshold_fraction) * 4.0 + 0.35)

    def to_dict(self) -> dict:
        return {
            "species": self.species,
            "primary_strength": self.primary_strength,
            "reproduction_energy_fraction": self.reproduction_energy_fraction,
            "maturity_fraction": self.maturity_fraction,
            "decisions": self.decisions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InstinctController":
        values = dict(data)
        values.setdefault("reproduction_energy_fraction", 0.7)
        values.setdefault("maturity_fraction", 0.02)
        return cls(**values)


@dataclass(slots=True)
class ActionArbiter:
    """Mix innate logits with an individual's learned residual logits."""

    instinct_weight: float = 1.0
    adaptive_weight: float = 0.12
    temperature: float = 0.85
    exploration: float = 0.035
    decisions: int = 0
    last_probabilities: list[float] | None = field(default=None, repr=False)
    last_instinct_preferences: list[float] | None = field(default=None, repr=False)
    last_adaptive_preferences: list[float] | None = field(default=None, repr=False)
    last_actions: list[int] | None = field(default=None, repr=False)

    def probabilities(
        self,
        instinct_preferences: list[float],
        adaptive_preferences: list[float],
        adaptive_enabled: bool,
    ) -> list[float]:
        adaptive_weight = self.adaptive_weight if adaptive_enabled else 0.0
        probabilities: list[float] = []
        offset = 0
        for size in HEAD_SIZES:
            logits = [
                (self.instinct_weight * instinct_preferences[offset + index]
                 + adaptive_weight * adaptive_preferences[offset + index]) / self.temperature
                for index in range(size)
            ]
            peak = max(logits)
            exponentials = [math.exp(max(-30.0, value - peak)) for value in logits]
            total = sum(exponentials)
            probabilities.extend(
                (1.0 - self.exploration) * value / total + self.exploration / size
                for value in exponentials
            )
            offset += size
        return probabilities

    def choose(
        self,
        instinct_preferences: list[float],
        adaptive_preferences: list[float],
        adaptive_enabled: bool,
        rng: random.Random,
    ) -> tuple[EmbodiedAction, list[float]]:
        probabilities = self.probabilities(
            instinct_preferences, adaptive_preferences, adaptive_enabled
        )
        actions: list[int] = []
        offset = 0
        for size in HEAD_SIZES:
            pick = rng.random()
            cumulative = 0.0
            action = size - 1
            for index in range(size):
                cumulative += probabilities[offset + index]
                if pick <= cumulative:
                    action = index
                    break
            actions.append(action)
            offset += size
        self.decisions += 1
        self.last_probabilities = probabilities[:]
        self.last_instinct_preferences = instinct_preferences[:]
        self.last_adaptive_preferences = adaptive_preferences[:]
        self.last_actions = actions[:]
        return EmbodiedAction.from_indices(actions), probabilities

    def to_dict(self) -> dict:
        return {
            "instinct_weight": self.instinct_weight,
            "adaptive_weight": self.adaptive_weight,
            "temperature": self.temperature,
            "exploration": self.exploration,
            "decisions": self.decisions,
            "last_probabilities": self.last_probabilities,
            "last_instinct_preferences": self.last_instinct_preferences,
            "last_adaptive_preferences": self.last_adaptive_preferences,
            "last_actions": self.last_actions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionArbiter":
        values = dict(data)
        for name in (
            "last_probabilities",
            "last_instinct_preferences",
            "last_adaptive_preferences",
            "last_actions",
        ):
            if values.get(name) is not None:
                values[name] = values[name][:]
        return cls(**values)
