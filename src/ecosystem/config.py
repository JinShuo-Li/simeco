"""Simulation parameters collected in serializable dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class SpeciesConfig:
    initial_count: int
    initial_energy: float
    max_energy: float
    move_cost: float
    idle_cost: float
    reproduce_energy: float
    reproduce_cost: float
    maturity_age: int
    max_age: int
    vision: int
    reproduction_chance: float
    hidden_size: int = 10
    learning_rate: float = 0.035
    mutation_rate: float = 0.08
    mutation_scale: float = 0.12


@dataclass(slots=True)
class WorldConfig:
    width: int = 48
    height: int = 22
    plant_capacity: float = 10.0
    initial_plant_fraction: float = 0.62
    plant_growth: float = 0.07
    plant_spread: float = 0.025
    plant_bite: float = 3.2
    plant_energy: float = 2.7
    prey_energy_fraction: float = 0.50
    capture_probability: float = 0.40
    crowding_cost: float = 0.10
    history_interval: int = 10
    herbivore: SpeciesConfig = field(
        default_factory=lambda: SpeciesConfig(
            initial_count=60,
            initial_energy=18.0,
            max_energy=34.0,
            move_cost=0.34,
            idle_cost=0.16,
            reproduce_energy=25.0,
            reproduce_cost=11.5,
            maturity_age=18,
            max_age=620,
            vision=4,
            reproduction_chance=0.004,
            learning_rate=0.030,
        )
    )
    predator: SpeciesConfig = field(
        default_factory=lambda: SpeciesConfig(
            initial_count=4,
            initial_energy=30.0,
            max_energy=55.0,
            move_cost=0.08,
            idle_cost=0.05,
            reproduce_energy=36.0,
            reproduce_cost=15.0,
            maturity_age=50,
            max_age=3000,
            vision=6,
            reproduction_chance=0.0015,
            learning_rate=0.015,
        )
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorldConfig":
        values = dict(data)
        values["herbivore"] = SpeciesConfig(**values["herbivore"])
        values["predator"] = SpeciesConfig(**values["predator"])
        return cls(**values)
