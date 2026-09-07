"""Innate behavior and arbitration between instinct and adaptive preferences."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass(slots=True)
class InstinctController:
    """Species-specific survival behavior available unchanged from birth."""

    species: str
    decisions: int = 0

    def preferences(self, observation: list[float]) -> list[float]:
        energy = observation[1]
        plant_here = observation[3]
        plants = observation[4:8]
        prey = observation[8:12]
        predators = observation[12:16]
        scores = [0.0] * 5

        if self.species == "herbivore":
            # Rest and eat when standing on useful food, especially when hungry.
            scores[0] += plant_here * (1.2 + 2.4 * (1.0 - energy))
            scores[0] += 0.35 * energy
            best_food = max(plants)
            if best_food > 0.0:
                scores[plants.index(best_food) + 1] += 1.8 * best_food

            # Escape in the direction opposite the strongest predator signal.
            danger = max(predators)
            if danger > 0.0:
                threat_action = predators.index(danger) + 1
                safe_action = {1: 3, 2: 4, 3: 1, 4: 2}[threat_action]
                scores[safe_action] += 4.5 * danger
                scores[threat_action] -= 3.2 * danger
                scores[0] -= 1.5 * danger
        elif self.species == "predator":
            target = max(prey)
            if target > 0.0:
                pursuit_action = prey.index(target) + 1
                scores[pursuit_action] += 4.2 * target
                scores[0] -= 1.2 * target
            else:
                # Conserve energy when satiated; hungry animals search stochastically.
                scores[0] += 2.0 * energy
                for action in range(1, 5):
                    scores[action] += 0.35 * (1.0 - energy)
        else:
            raise ValueError(f"unknown species: {self.species}")
        return scores

    def reproduction_tendency(
        self, observation: list[float], maturity_fraction: float, threshold_fraction: float
    ) -> float:
        """Innate gate for energy- and maturity-dependent reproduction."""
        if observation[1] < threshold_fraction or observation[2] < maturity_fraction:
            return 0.0
        return min(1.0, (observation[1] - threshold_fraction) * 4.0 + 0.35)

    def to_dict(self) -> dict:
        return {"species": self.species, "decisions": self.decisions}

    @classmethod
    def from_dict(cls, data: dict) -> "InstinctController":
        return cls(**data)


@dataclass(slots=True)
class ActionArbiter:
    """Mix innate logits with an individual's learned residual logits."""

    instinct_weight: float = 1.0
    adaptive_weight: float = 0.40
    temperature: float = 0.85
    exploration: float = 0.035
    decisions: int = 0
    last_probabilities: list[float] | None = field(default=None, repr=False)
    last_instinct_preferences: list[float] | None = field(default=None, repr=False)
    last_adaptive_preferences: list[float] | None = field(default=None, repr=False)

    def probabilities(
        self,
        instinct_preferences: list[float],
        adaptive_preferences: list[float],
        adaptive_enabled: bool,
    ) -> list[float]:
        adaptive_weight = self.adaptive_weight if adaptive_enabled else 0.0
        logits = [
            (self.instinct_weight * instinct + adaptive_weight * adaptive) / self.temperature
            for instinct, adaptive in zip(instinct_preferences, adaptive_preferences)
        ]
        peak = max(logits)
        exponentials = [math.exp(max(-30.0, value - peak)) for value in logits]
        total = sum(exponentials)
        policy = [value / total for value in exponentials]
        return [
            (1.0 - self.exploration) * value + self.exploration / len(policy)
            for value in policy
        ]

    def choose(
        self,
        instinct_preferences: list[float],
        adaptive_preferences: list[float],
        adaptive_enabled: bool,
        rng: random.Random,
    ) -> tuple[int, list[float]]:
        probabilities = self.probabilities(
            instinct_preferences, adaptive_preferences, adaptive_enabled
        )
        pick = rng.random()
        cumulative = 0.0
        action = len(probabilities) - 1
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if pick <= cumulative:
                action = index
                break
        self.decisions += 1
        self.last_probabilities = probabilities[:]
        self.last_instinct_preferences = instinct_preferences[:]
        self.last_adaptive_preferences = adaptive_preferences[:]
        return action, probabilities

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
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionArbiter":
        values = dict(data)
        for name in (
            "last_probabilities",
            "last_instinct_preferences",
            "last_adaptive_preferences",
        ):
            if values.get(name) is not None:
                values[name] = values[name][:]
        return cls(**values)
