"""Serializable organisms and run metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

from .controllers import ActionArbiter, InstinctController
from .network import AdaptivePolicy


@dataclass(slots=True)
class Organism:
    id: int
    species: str
    x: int
    y: int
    energy: float
    age: int
    generation: int
    parent_id: int | None
    heading: int
    instinct: InstinctController
    adaptive_policy: AdaptivePolicy
    arbiter: ActionArbiter
    lifetime_reward: float = 0.0
    meals: int = 0
    offspring_count: int = 0
    reproduction_progress: float = 0.0
    action_counts: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "species": self.species,
            "x": self.x,
            "y": self.y,
            "energy": self.energy,
            "age": self.age,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "heading": self.heading,
            "instinct": self.instinct.to_dict(),
            "adaptive_policy": self.adaptive_policy.to_dict(),
            "arbiter": self.arbiter.to_dict(),
            "lifetime_reward": self.lifetime_reward,
            "meals": self.meals,
            "offspring_count": self.offspring_count,
            "reproduction_progress": self.reproduction_progress,
            "action_counts": self.action_counts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Organism":
        values = dict(data)
        # V1 snapshots stored only the individual network as ``policy``.
        policy_data = values.pop("policy", None)
        values["adaptive_policy"] = AdaptivePolicy.from_dict(
            values.get("adaptive_policy", policy_data)
        )
        values["instinct"] = InstinctController.from_dict(
            values.get("instinct", {"species": values["species"]})
        )
        values["arbiter"] = ActionArbiter.from_dict(values.get("arbiter", {}))
        values.setdefault("action_counts", [0, 0, 0, 0])
        values.setdefault("reproduction_progress", 0.0)
        values.setdefault("heading", 0)
        return cls(**values)

    @property
    def policy(self) -> AdaptivePolicy:
        """Compatibility alias for V1 analysis scripts."""
        return self.adaptive_policy


@dataclass(slots=True)
class Metrics:
    births_herbivore: int = 0
    births_predator: int = 0
    deaths_starvation: int = 0
    deaths_age: int = 0
    deaths_predation: int = 0
    hunts: int = 0
    hunt_attempts: int = 0
    prey_escapes: int = 0
    plants_eaten: float = 0.0
    reward_herbivore: float = 0.0
    reward_predator: float = 0.0
    learning_updates: int = 0
    actions_herbivore: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    actions_predator: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    efforts_herbivore: list[int] = field(default_factory=lambda: [0, 0, 0])
    efforts_predator: list[int] = field(default_factory=lambda: [0, 0, 0])
    interactions_herbivore: list[int] = field(default_factory=lambda: [0, 0, 0])
    interactions_predator: list[int] = field(default_factory=lambda: [0, 0, 0])
    reproduction_intents_herbivore: int = 0
    reproduction_intents_predator: int = 0
    energy_spent_herbivore: float = 0.0
    energy_spent_predator: float = 0.0
    energy_gained_herbivore: float = 0.0
    energy_gained_predator: float = 0.0
    unnecessary_sprints: int = 0
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "births_herbivore": self.births_herbivore,
            "births_predator": self.births_predator,
            "deaths_starvation": self.deaths_starvation,
            "deaths_age": self.deaths_age,
            "deaths_predation": self.deaths_predation,
            "hunts": self.hunts,
            "hunt_attempts": self.hunt_attempts,
            "prey_escapes": self.prey_escapes,
            "plants_eaten": self.plants_eaten,
            "reward_herbivore": self.reward_herbivore,
            "reward_predator": self.reward_predator,
            "learning_updates": self.learning_updates,
            "actions_herbivore": self.actions_herbivore,
            "actions_predator": self.actions_predator,
            "efforts_herbivore": self.efforts_herbivore,
            "efforts_predator": self.efforts_predator,
            "interactions_herbivore": self.interactions_herbivore,
            "interactions_predator": self.interactions_predator,
            "reproduction_intents_herbivore": self.reproduction_intents_herbivore,
            "reproduction_intents_predator": self.reproduction_intents_predator,
            "energy_spent_herbivore": self.energy_spent_herbivore,
            "energy_spent_predator": self.energy_spent_predator,
            "energy_gained_herbivore": self.energy_gained_herbivore,
            "energy_gained_predator": self.energy_gained_predator,
            "unnecessary_sprints": self.unnecessary_sprints,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Metrics":
        values = dict(data)
        values.setdefault("prey_escapes", 0)
        values.setdefault("actions_herbivore", [0, 0, 0, 0])
        values.setdefault("actions_predator", [0, 0, 0, 0])
        values.setdefault("efforts_herbivore", [0, 0, 0])
        values.setdefault("efforts_predator", [0, 0, 0])
        values.setdefault("interactions_herbivore", [0, 0, 0])
        values.setdefault("interactions_predator", [0, 0, 0])
        values.setdefault("reproduction_intents_herbivore", 0)
        values.setdefault("reproduction_intents_predator", 0)
        values.setdefault("energy_spent_herbivore", 0.0)
        values.setdefault("energy_spent_predator", 0.0)
        values.setdefault("energy_gained_herbivore", 0.0)
        values.setdefault("energy_gained_predator", 0.0)
        values.setdefault("unnecessary_sprints", 0)
        return cls(**values)
