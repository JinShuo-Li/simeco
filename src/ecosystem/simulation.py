"""Spatial resource, herbivore, and predator simulation."""

from __future__ import annotations

import base64
import pickle
import random
from collections import defaultdict

from .config import SpeciesConfig, WorldConfig
from .controllers import ActionArbiter, InstinctController
from .model import Metrics, Organism
from .network import AdaptivePolicy

ACTIONS = ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0))
OBSERVATION_SIZE = 16


class Simulation:
    def __init__(self, config: WorldConfig | None = None, seed: int = 1, learning: bool = True):
        self.config = config or WorldConfig()
        self.seed = seed
        self.learning = learning
        self.rng = random.Random(seed)
        self.step_count = 0
        self.next_id = 1
        self.resources = self._initial_resources()
        self.organisms: dict[int, Organism] = {}
        self.metrics = Metrics()
        self.last_events = {"births": 0, "deaths": 0, "hunts": 0}
        self._spawn_initial("herbivore", self.config.herbivore)
        self._spawn_initial("predator", self.config.predator)
        self._record_history()

    def _initial_resources(self) -> list[list[float]]:
        cfg = self.config
        return [
            [
                cfg.plant_capacity * max(0.05, min(1.0, self.rng.gauss(cfg.initial_plant_fraction, 0.20)))
                for _ in range(cfg.width)
            ]
            for _ in range(cfg.height)
        ]

    def _spawn_initial(self, species: str, cfg: SpeciesConfig) -> None:
        for _ in range(cfg.initial_count):
            policy = AdaptivePolicy.random(OBSERVATION_SIZE, cfg.hidden_size, len(ACTIONS), self.rng)
            organism = Organism(
                id=self.next_id,
                species=species,
                x=self.rng.randrange(self.config.width),
                y=self.rng.randrange(self.config.height),
                energy=cfg.initial_energy * self.rng.uniform(0.75, 1.15),
                age=self.rng.randrange(max(1, cfg.maturity_age)),
                generation=0,
                parent_id=None,
                instinct=InstinctController(species),
                adaptive_policy=policy,
                arbiter=ActionArbiter(),
            )
            self.organisms[organism.id] = organism
            self.next_id += 1

    def species(self, name: str) -> list[Organism]:
        return [organism for organism in self.organisms.values() if organism.species == name]

    def _distance_vector(self, source: Organism, targets: list[Organism], vision: int) -> list[float]:
        scores = [0.0, 0.0, 0.0, 0.0]
        width, height = self.config.width, self.config.height
        for target in targets:
            if target.id == source.id:
                continue
            dx = (target.x - source.x + width // 2) % width - width // 2
            dy = (target.y - source.y + height // 2) % height - height // 2
            distance = abs(dx) + abs(dy)
            if not 0 < distance <= vision:
                continue
            strength = (vision + 1 - distance) / vision
            if abs(dx) >= abs(dy) and dx:
                scores[1 if dx > 0 else 3] += strength
            if abs(dy) >= abs(dx) and dy:
                scores[2 if dy > 0 else 0] += strength
        return [min(1.0, value) for value in scores]

    def observe(self, organism: Organism, herbivores: list[Organism], predators: list[Organism]) -> list[float]:
        cfg = self.config.herbivore if organism.species == "herbivore" else self.config.predator
        plant_directions = []
        for dx, dy in ACTIONS[1:]:
            total = 0.0
            for distance in range(1, cfg.vision + 1):
                x = (organism.x + dx * distance) % self.config.width
                y = (organism.y + dy * distance) % self.config.height
                total += self.resources[y][x] / (self.config.plant_capacity * distance)
            plant_directions.append(min(1.0, total / 1.6))
        prey_signal = self._distance_vector(organism, herbivores, cfg.vision)
        predator_signal = self._distance_vector(organism, predators, cfg.vision)
        return [
            1.0,
            min(1.0, organism.energy / cfg.max_energy),
            min(1.0, organism.age / cfg.max_age),
            self.resources[organism.y][organism.x] / self.config.plant_capacity,
            *plant_directions,
            *prey_signal,
            *predator_signal,
        ]

    def _grow_resources(self) -> None:
        cfg = self.config
        old = self.resources
        new = [[0.0] * cfg.width for _ in range(cfg.height)]
        for y in range(cfg.height):
            for x in range(cfg.width):
                biomass = old[y][x]
                logistic = cfg.plant_growth * biomass * (1.0 - biomass / cfg.plant_capacity)
                neighbors = (
                    old[(y - 1) % cfg.height][x]
                    + old[(y + 1) % cfg.height][x]
                    + old[y][(x - 1) % cfg.width]
                    + old[y][(x + 1) % cfg.width]
                ) / 4.0
                spread = cfg.plant_spread * (neighbors - biomass)
                new[y][x] = max(0.0, min(cfg.plant_capacity, biomass + logistic + spread))
        self.resources = new

    def step(self) -> None:
        self.step_count += 1
        self.last_events = {"births": 0, "deaths": 0, "hunts": 0}
        self._grow_resources()
        herbivores = self.species("herbivore")
        predators = self.species("predator")
        order = list(self.organisms.values())
        self.rng.shuffle(order)
        occupied: dict[tuple[int, int], int] = defaultdict(int)
        for animal in order:
            occupied[(animal.x, animal.y)] += 1

        rewards: dict[int, float] = {}
        for animal in order:
            cfg = self.config.herbivore if animal.species == "herbivore" else self.config.predator
            observation = self.observe(animal, herbivores, predators)
            instinct_preferences = animal.instinct.preferences(observation)
            adaptive_preferences = animal.adaptive_policy.preferences(observation)
            action, combined_probabilities = animal.arbiter.choose(
                instinct_preferences,
                adaptive_preferences,
                adaptive_enabled=self.learning,
                rng=self.rng,
            )
            animal.instinct.decisions += 1
            if self.learning:
                animal.adaptive_policy.record_decision(
                    observation, action, combined_probabilities
                )
            animal.action_counts[action] += 1
            dx, dy = ACTIONS[action]
            occupied[(animal.x, animal.y)] -= 1
            animal.x = (animal.x + dx) % self.config.width
            animal.y = (animal.y + dy) % self.config.height
            occupied[(animal.x, animal.y)] += 1
            cost = cfg.idle_cost if action == 0 else cfg.move_cost
            cost += self.config.crowding_cost * max(0, occupied[(animal.x, animal.y)] - 2)
            animal.energy -= cost
            reward = -cost / cfg.move_cost * 0.08
            if animal.species == "herbivore":
                available = self.resources[animal.y][animal.x]
                eaten = min(available, self.config.plant_bite)
                self.resources[animal.y][animal.x] -= eaten
                gained = eaten * self.config.plant_energy
                animal.energy = min(cfg.max_energy, animal.energy + gained)
                self.metrics.plants_eaten += eaten
                if eaten > 0.25:
                    animal.meals += 1
                    reward += gained / 4.0
            reward += 0.01  # surviving another tick is weak positive feedback
            rewards[animal.id] = reward

        self._resolve_hunts(predators, rewards)
        newborns: list[Organism] = []
        dead: list[int] = []
        for animal in order:
            if animal.id not in self.organisms:
                continue
            cfg = self.config.herbivore if animal.species == "herbivore" else self.config.predator
            animal.age += 1
            reward = rewards.get(animal.id, 0.0)
            if animal.energy <= 0:
                reward -= 3.0
                self.metrics.deaths_starvation += 1
                dead.append(animal.id)
            elif animal.age >= cfg.max_age:
                self.metrics.deaths_age += 1
                dead.append(animal.id)
            else:
                current_observation = self.observe(animal, herbivores, predators)
                reproduction_drive = animal.instinct.reproduction_tendency(
                    current_observation,
                    cfg.maturity_age / cfg.max_age,
                    cfg.reproduce_energy / cfg.max_energy,
                )
            if (
                animal.id not in dead
                and reproduction_drive > 0.0
                and self.rng.random() < cfg.reproduction_chance * reproduction_drive
            ):
                animal.energy -= cfg.reproduce_cost
                child_policy = animal.adaptive_policy.offspring(
                    self.rng, cfg.mutation_rate, cfg.mutation_scale
                )
                child = Organism(
                    id=self.next_id,
                    species=animal.species,
                    x=(animal.x + self.rng.choice((-1, 0, 1))) % self.config.width,
                    y=(animal.y + self.rng.choice((-1, 0, 1))) % self.config.height,
                    energy=cfg.reproduce_cost * 0.78,
                    age=0,
                    generation=animal.generation + 1,
                    parent_id=animal.id,
                    instinct=InstinctController(animal.species),
                    adaptive_policy=child_policy,
                    arbiter=ActionArbiter(),
                )
                self.next_id += 1
                newborns.append(child)
                animal.offspring_count += 1
                reward += 1.3
                if animal.species == "herbivore":
                    self.metrics.births_herbivore += 1
                else:
                    self.metrics.births_predator += 1
            animal.lifetime_reward += reward
            if self.learning:
                animal.adaptive_policy.learn(reward, cfg.learning_rate)
                self.metrics.learning_updates += 1
            if animal.species == "herbivore":
                self.metrics.reward_herbivore += reward
            else:
                self.metrics.reward_predator += reward

        for animal_id in dead:
            self.organisms.pop(animal_id, None)
        for child in newborns:
            self.organisms[child.id] = child
        self.last_events["births"] = len(newborns)
        self.last_events["deaths"] += len(dead)
        if self.step_count % self.config.history_interval == 0:
            self._record_history()

    def _resolve_hunts(self, predators: list[Organism], rewards: dict[int, float]) -> None:
        prey_by_cell: dict[tuple[int, int], list[Organism]] = defaultdict(list)
        for prey in self.species("herbivore"):
            prey_by_cell[(prey.x, prey.y)].append(prey)
        for predator in predators:
            if predator.id not in self.organisms:
                continue
            candidates = prey_by_cell.get((predator.x, predator.y), [])
            if not candidates:
                rewards[predator.id] = rewards.get(predator.id, 0.0) - 0.07
                continue
            self.metrics.hunt_attempts += 1
            prey = self.rng.choice(candidates)
            if self.rng.random() > self.config.capture_probability:
                rewards[predator.id] = rewards.get(predator.id, 0.0) - 0.25
                rewards[prey.id] = rewards.get(prey.id, 0.0) + 0.45
                self.metrics.prey_escapes += 1
                continue
            candidates.remove(prey)
            if prey.id not in self.organisms:
                continue
            self.organisms.pop(prey.id)
            gain = max(3.0, prey.energy * self.config.prey_energy_fraction)
            predator.energy = min(self.config.predator.max_energy, predator.energy + gain)
            predator.meals += 1
            rewards[predator.id] = rewards.get(predator.id, 0.0) + 3.2 + gain / 10.0
            rewards[prey.id] = rewards.get(prey.id, 0.0) - 4.0
            self.metrics.hunts += 1
            self.metrics.deaths_predation += 1
            self.last_events["hunts"] += 1
            self.last_events["deaths"] += 1

    def _record_history(self) -> None:
        herbivores = self.species("herbivore")
        predators = self.species("predator")
        count = self.config.width * self.config.height
        self.metrics.history.append(
            {
                "step": self.step_count,
                "herbivores": len(herbivores),
                "predators": len(predators),
                "plants": sum(map(sum, self.resources)) / count,
                "herbivore_energy": sum(a.energy for a in herbivores) / max(1, len(herbivores)),
                "predator_energy": sum(a.energy for a in predators) / max(1, len(predators)),
                "max_generation": max((a.generation for a in self.organisms.values()), default=0),
                "hunts": self.metrics.hunts,
            }
        )

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def rng_state(self) -> str:
        return base64.b85encode(pickle.dumps(self.rng.getstate())).decode("ascii")

    def set_rng_state(self, encoded: str) -> None:
        self.rng.setstate(pickle.loads(base64.b85decode(encoded.encode("ascii"))))
