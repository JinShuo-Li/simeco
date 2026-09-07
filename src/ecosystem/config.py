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
    hidden_size: int = 10
    learning_rate: float = 0.035
    mutation_rate: float = 0.08
    mutation_scale: float = 0.12


@dataclass(slots=True)
class WorldConfig:
    width: int = 48
    height: int = 22
    plant_capacity: float = 12.0
    initial_plant_fraction: float = 0.72
    plant_growth: float = 0.032
    plant_spread: float = 0.025
    plant_bite: float = 1.8
    plant_energy: float = 1.05
    prey_energy_fraction: float = 0.72
    crowding_cost: float = 0.055
    reproduction_chance: float = 0.045
    history_interval: int = 10
    herbivore: SpeciesConfig = field(
        default_factory=lambda: SpeciesConfig(
            initial_count=105,
            initial_energy=11.0,
            max_energy=24.0,
            move_cost=0.30,
            idle_cost=0.20,
            reproduce_energy=17.0,
            reproduce_cost=7.0,
            maturity_age=25,
            max_age=340,
            vision=4,
            learning_rate=0.030,
        )
    )
    predator: SpeciesConfig = field(
        default_factory=lambda: SpeciesConfig(
            initial_count=18,
            initial_energy=18.0,
            max_energy=42.0,
            move_cost=0.38,
            idle_cost=0.29,
            reproduce_energy=32.0,
            reproduce_cost=13.0,
            maturity_age=35,
            max_age=430,
            vision=6,
            learning_rate=0.040,
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
    plant_capacity: float = 10.0
    plant_growth: float = 0.07
    plant_spread: float = 0.025
    initial_plant_fraction: float = 0.62
    herbivore: SpeciesConfig = field(
        default_factory=lambda: SpeciesConfig(
            initial_count=105,
            initial_energy=18.0,
            max_energy=34.0,
            move_cost=0.34,
            idle_cost=0.16,
            reproduce_energy=25.0,
            reproduce_cost=11.5,
            maturity_age=18,
            max_age=620,
            vision=4,
        )
    )
    predator: SpeciesConfig = field(
        default_factory=lambda: SpeciesConfig(
            initial_count=18,
            initial_energy=30.0,
            max_energy=55.0,
            move_cost=0.48,
            idle_cost=0.26,
            reproduce_energy=43.0,
            reproduce_cost=20.0,
            maturity_age=28,
            max_age=760,
            vision=5,
            learning_rate=0.045,
        )
    )
    plant_bite: float = 3.2
    plant_energy: float = 2.7
    prey_energy_fraction: float = 0.72
    crowding_cost: float = 0.10
    reproduction_chance: float = 0.035
    history_interval: int = 10

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorldConfig":
        data = dict(data)
        data["herbivore"] = SpeciesConfig(**data["herbivore"])
        data["predator"] = SpeciesConfig(**data["predator"])
        return cls(**data)
