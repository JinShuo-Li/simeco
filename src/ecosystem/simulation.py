"""Spatial resource, herbivore, and predator simulation."""

from __future__ import annotations

import base64
import pickle
import random
from collections import defaultdict

from .actions import (
    HEAD_SIZES,
    TOTAL_ACTION_OUTPUTS,
    Effort,
    EmbodiedAction,
    Interaction,
    Locomotion,
    Reproduction,
)
from .config import SpeciesConfig, WorldConfig
from .controllers import ActionArbiter, InstinctController
from .model import Metrics, Organism
from .network import AdaptivePolicy
from .perception import OBSERVATION_SIZE, EgocentricPerception, channel_index
from .social import visible_individuals

HEADINGS = ((0, -1), (1, 0), (0, 1), (-1, 0))


class Simulation:
    def __init__(
        self,
        config: WorldConfig | None = None,
        seed: int = 1,
        learning: bool = True,
        memory: bool | None = None,
        social_memory: bool | None = None,
    ):
        self.config = config or WorldConfig()
        self.seed = seed
        self.learning = learning
        self.memory = learning if memory is None else learning and memory
        self.social_memory = (
            self.memory if social_memory is None else self.memory and social_memory
        )
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
            policy = AdaptivePolicy.random(
                OBSERVATION_SIZE,
                cfg.hidden_size,
                TOTAL_ACTION_OUTPUTS,
                self.rng,
                memory_size=cfg.memory_size,
            )
            organism = Organism(
                id=self.next_id,
                species=species,
                x=self.rng.randrange(self.config.width),
                y=self.rng.randrange(self.config.height),
                energy=cfg.initial_energy * self.rng.uniform(0.75, 1.15),
                age=self.rng.randrange(max(1, cfg.maturity_age)),
                generation=0,
                parent_id=None,
                heading=self.rng.randrange(4),
                instinct=InstinctController(
                    species,
                    cfg.instinct_strength,
                    cfg.reproduce_energy / cfg.max_energy,
                    cfg.maturity_age / cfg.max_age,
                ),
                adaptive_policy=policy,
                arbiter=ActionArbiter(),
                reproduction_progress=self.rng.random(),
            )
            self.organisms[organism.id] = organism
            self.next_id += 1

    def species(self, name: str) -> list[Organism]:
        return [organism for organism in self.organisms.values() if organism.species == name]

    @property
    def controller_mode(self) -> str:
        if not self.learning:
            return "instinct_only"
        if not self.memory:
            return "instinct+learning"
        return "instinct+learning+memory+social" if self.social_memory else "instinct+learning+memory"

    def observe(self, organism: Organism, herbivores: list[Organism], predators: list[Organism]) -> list[float]:
        cfg = self.config.herbivore if organism.species == "herbivore" else self.config.predator
        return EgocentricPerception.encode(
            self.config, cfg, self.resources, organism, herbivores, predators
        )

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
        head_rewards: dict[int, list[float]] = {}
        decisions: dict[int, EmbodiedAction] = {}
        for animal in order:
            cfg = self.config.herbivore if animal.species == "herbivore" else self.config.predator
            observation = self.observe(animal, herbivores, predators)
            instinct_preferences = animal.instinct.preferences(observation)
            previous_action = animal.arbiter.last_actions
            social_slots = visible_individuals(animal, order, self.config, cfg)
            adaptive_preferences = (
                animal.adaptive_policy.advance(
                    observation,
                    use_memory=self.memory,
                    social_slots=social_slots if self.social_memory else None,
                    social_enabled=self.social_memory,
                    step=self.step_count,
                )
                if self.learning
                else [0.0] * TOTAL_ACTION_OUTPUTS
            )
            action, combined_probabilities = animal.arbiter.choose(
                instinct_preferences,
                adaptive_preferences,
                adaptive_enabled=self.learning,
                rng=self.rng,
            )
            decisions[animal.id] = action
            visible_ids = [slot["id"] for slot in social_slots]
            previous_visible = set(animal.visible_ids_last_tick)
            self.metrics.social_encounters += len(visible_ids)
            self.metrics.repeated_social_encounters += sum(
                identity in previous_visible for identity in visible_ids
            )
            same_species = [
                slot for slot in social_slots
                if self.organisms[slot["id"]].species == animal.species
            ]
            self.metrics.same_species_encounters += len(same_species)
            if any(slot["features"][0] > 0.0 for slot in same_species):
                self.metrics.follow_opportunities += 1
                if action.locomotion == Locomotion.FORWARD:
                    self.metrics.follow_actions += 1
            if animal.species == "predator":
                colocated = [
                    slot["id"] for slot in same_species if slot["features"][2] == 0.0
                ]
                self.metrics.predator_colocations += len(colocated)
                self.metrics.repeated_predator_colocations += sum(
                    identity in previous_visible for identity in colocated
                )
            animal.visible_ids_last_tick = visible_ids
            if previous_action is not None:
                self.metrics.temporal_action_pairs += 1
                if previous_action[0] == action.locomotion:
                    self.metrics.locomotion_repeats += 1
                if previous_action[1] == action.effort:
                    self.metrics.effort_repeats += 1
            animal.instinct.decisions += 1
            if self.learning:
                animal.adaptive_policy.record_decision(
                    observation,
                    action,
                    combined_probabilities,
                    animal.arbiter.adaptive_weight / animal.arbiter.temperature,
                )
            animal.action_counts[action.locomotion] += 1
            action_metric = (
                self.metrics.actions_herbivore
                if animal.species == "herbivore"
                else self.metrics.actions_predator
            )
            action_metric[action.locomotion] += 1
            effort_metric = (
                self.metrics.efforts_herbivore
                if animal.species == "herbivore"
                else self.metrics.efforts_predator
            )
            interaction_metric = (
                self.metrics.interactions_herbivore
                if animal.species == "herbivore"
                else self.metrics.interactions_predator
            )
            effort_metric[action.effort] += 1
            interaction_metric[action.interaction] += 1
            if action.reproduction == Reproduction.INTEND:
                if animal.species == "herbivore":
                    self.metrics.reproduction_intents_herbivore += 1
                else:
                    self.metrics.reproduction_intents_predator += 1
            occupied[(animal.x, animal.y)] -= 1
            if action.locomotion == Locomotion.TURN_LEFT:
                animal.heading = (animal.heading - 1) % 4
            elif action.locomotion == Locomotion.TURN_RIGHT:
                animal.heading = (animal.heading + 1) % 4
            if action.locomotion == Locomotion.FORWARD or (
                action.locomotion in (Locomotion.TURN_LEFT, Locomotion.TURN_RIGHT)
                and action.effort != Effort.LOW
            ):
                dx, dy = HEADINGS[animal.heading]
                distance = 2 if action.effort == Effort.SPRINT else 1
                animal.x = (animal.x + dx * distance) % self.config.width
                animal.y = (animal.y + dy * distance) % self.config.height
            occupied[(animal.x, animal.y)] += 1

            if action.locomotion == Locomotion.HOLD:
                cost = cfg.idle_cost * (0.65, 1.0, 1.5)[action.effort]
            elif action.locomotion == Locomotion.FORWARD:
                cost = cfg.move_cost * (0.65, 1.0, 2.4)[action.effort]
            else:
                cost = cfg.move_cost * (0.35, 0.55, 0.9)[action.effort]
            if action.interaction == Interaction.ATTACK:
                cost += cfg.move_cost * 0.8
            cost += self.config.crowding_cost * max(0, occupied[(animal.x, animal.y)] - 2)
            if animal.species == "predator":
                cost += cfg.competition_cost * sum(
                    observation[channel_index("predators", forward, lateral)]
                    for forward in (-1, 0, 1)
                    for lateral in (-1, 0, 1)
                )
            if action.effort == Effort.SPRINT:
                stimulus_channel = "predators" if animal.species == "herbivore" else "herbivores"
                stimulus = sum(
                    observation[channel_index(stimulus_channel, forward, lateral)]
                    for forward in (-1, 0, 1)
                    for lateral in (-1, 0, 1)
                )
                if stimulus == 0.0:
                    self.metrics.unnecessary_sprints += 1
            animal.energy -= cost
            if animal.species == "herbivore":
                self.metrics.energy_spent_herbivore += cost
            else:
                self.metrics.energy_spent_predator += cost
            reward = -cost / cfg.move_cost * 0.08
            per_head = [reward, reward, 0.01, 0.01]
            if animal.species == "herbivore" and action.interaction == Interaction.FEED:
                available = self.resources[animal.y][animal.x]
                eaten = min(available, self.config.plant_bite)
                self.resources[animal.y][animal.x] -= eaten
                gained = eaten * self.config.plant_energy
                animal.energy = min(cfg.max_energy, animal.energy + gained)
                self.metrics.energy_gained_herbivore += gained
                self.metrics.plants_eaten += eaten
                if eaten > 0.25:
                    animal.meals += 1
                    reward += gained / 4.0
                    per_head[0] += gained / 8.0
                    per_head[1] += gained / 10.0
                    per_head[2] += gained / 3.0
            reward += 0.01  # surviving another tick is weak positive feedback
            per_head[0] += 0.01
            per_head[1] += 0.01
            rewards[animal.id] = reward
            head_rewards[animal.id] = per_head

        self._resolve_hunts(predators, decisions, rewards, head_rewards)
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
                head_rewards[animal.id] = [
                    value - 3.0 for value in head_rewards.get(animal.id, [0.0] * 4)
                ]
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
                intent = decisions[animal.id].reproduction == Reproduction.INTEND
                if reproduction_drive > 0.0 and intent:
                    animal.reproduction_progress += cfg.reproduction_chance * reproduction_drive
                else:
                    animal.reproduction_progress *= 0.995
            if (
                animal.id not in dead
                and animal.reproduction_progress >= 1.0
            ):
                animal.reproduction_progress -= 1.0
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
                    heading=(animal.heading + self.rng.choice((-1, 0, 1))) % 4,
                    instinct=InstinctController(
                        animal.species,
                        cfg.instinct_strength,
                        cfg.reproduce_energy / cfg.max_energy,
                        cfg.maturity_age / cfg.max_age,
                    ),
                    adaptive_policy=child_policy,
                    arbiter=ActionArbiter(),
                    reproduction_progress=0.0,
                )
                self.next_id += 1
                newborns.append(child)
                animal.offspring_count += 1
                reward += 1.3
                head_rewards[animal.id][3] += 1.3
                if animal.species == "herbivore":
                    self.metrics.births_herbivore += 1
                else:
                    self.metrics.births_predator += 1
            animal.lifetime_reward += reward
            if self.learning:
                animal.adaptive_policy.learn(
                    reward,
                    cfg.learning_rate,
                    head_rewards.get(animal.id),
                    terminal=animal.id in dead,
                )
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

    def _resolve_hunts(
        self,
        predators: list[Organism],
        decisions: dict[int, EmbodiedAction],
        rewards: dict[int, float],
        head_rewards: dict[int, list[float]],
    ) -> None:
        prey_by_cell: dict[tuple[int, int], list[Organism]] = defaultdict(list)
        for prey in self.species("herbivore"):
            prey_by_cell[(prey.x, prey.y)].append(prey)
        for predator in predators:
            if predator.id not in self.organisms:
                continue
            if decisions[predator.id].interaction != Interaction.ATTACK:
                continue
            candidates = prey_by_cell.get((predator.x, predator.y), [])
            if not candidates:
                rewards[predator.id] = rewards.get(predator.id, 0.0) - 0.18
                head_rewards[predator.id][2] -= 0.18
                continue
            self.metrics.hunt_attempts += 1
            prey = self.rng.choice(candidates)
            if self.rng.random() > self.config.capture_probability:
                rewards[predator.id] = rewards.get(predator.id, 0.0) - 0.25
                rewards[prey.id] = rewards.get(prey.id, 0.0) + 0.45
                head_rewards[predator.id][2] -= 0.25
                self.metrics.prey_escapes += 1
                continue
            candidates.remove(prey)
            if prey.id not in self.organisms:
                continue
            self.organisms.pop(prey.id)
            gain = max(3.0, prey.energy * self.config.prey_energy_fraction)
            predator.energy = min(self.config.predator.max_energy, predator.energy + gain)
            self.metrics.energy_gained_predator += gain
            predator.meals += 1
            rewards[predator.id] = rewards.get(predator.id, 0.0) + 3.2 + gain / 10.0
            rewards[prey.id] = rewards.get(prey.id, 0.0) - 4.0
            head_rewards[predator.id][0] += gain / 15.0
            head_rewards[predator.id][1] += gain / 20.0
            head_rewards[predator.id][2] += 3.2 + gain / 10.0
            self.metrics.hunts += 1
            self.metrics.deaths_predation += 1
            self.last_events["hunts"] += 1
            self.last_events["deaths"] += 1

    def _record_history(self) -> None:
        herbivores = self.species("herbivore")
        predators = self.species("predator")
        count = self.config.width * self.config.height
        social_stats=[
            animal.adaptive_policy.social_statistics(self.step_count)
            for animal in self.organisms.values()
        ]
        mean_social=lambda name: sum(item[name] for item in social_stats)/max(1,len(social_stats))
        prey_actions=sum(self.metrics.actions_herbivore)
        predator_actions=sum(self.metrics.actions_predator)
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
                "starvation": self.metrics.deaths_starvation,
                "food_energy_efficiency": self.metrics.energy_gained_herbivore
                / max(.001,self.metrics.energy_spent_herbivore),
                "hunt_energy_efficiency": self.metrics.energy_gained_predator
                / max(.001,self.metrics.energy_spent_predator),
                "herbivore_reward_per_step": self.metrics.reward_herbivore/max(1,self.step_count),
                "predator_reward_per_step": self.metrics.reward_predator/max(1,self.step_count),
                "repeated_association": self.metrics.repeated_social_encounters
                / max(1,self.metrics.social_encounters),
                "following": self.metrics.follow_actions/max(1,self.metrics.follow_opportunities),
                "predator_colocation_persistence": self.metrics.repeated_predator_colocations
                / max(1,self.metrics.predator_colocations),
                "social_table_occupancy": mean_social("occupancy"),
                "known_individual_fraction": mean_social("known_fraction"),
                "social_eviction_rate": mean_social("eviction_rate"),
                "social_top3_concentration": mean_social("top3_concentration"),
                "social_dyad_streak": mean_social("mean_max_streak"),
                "social_distance": mean_social("mean_distance"),
                "herbivore_sprint_fraction": self.metrics.efforts_herbivore[Effort.SPRINT]
                / max(1,prey_actions),
                "predator_sprint_fraction": self.metrics.efforts_predator[Effort.SPRINT]
                / max(1,predator_actions),
            }
        )

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def rng_state(self) -> str:
        return base64.b85encode(pickle.dumps(self.rng.getstate())).decode("ascii")

    def set_rng_state(self, encoded: str) -> None:
        self.rng.setstate(pickle.loads(base64.b85decode(encoded.encode("ascii"))))
