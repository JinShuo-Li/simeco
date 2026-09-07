"""Serializable organisms and run metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

from .network import TinyMLP


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
    policy: TinyMLP
    lifetime_reward: float = 0.0
    meals: int = 0
    offspring_count: int = 0
    action_counts: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])

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
            "policy": self.policy.to_dict(),
            "lifetime_reward": self.lifetime_reward,
            "meals": self.meals,
            "offspring_count": self.offspring_count,
            "action_counts": self.action_counts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Organism":
        data = dict(data)
        data["policy"] = TinyMLP.from_dict(data["policy"])
        data.setdefault("action_counts", [0, 0, 0, 0, 0])
        return cls(**data)


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
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Metrics":
        values = dict(data)
        values.setdefault("prey_escapes", 0)
        return cls(**values)
