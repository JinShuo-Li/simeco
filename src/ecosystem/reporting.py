"""Metrics summaries and experiment log output."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .actions import Effort, Interaction, Locomotion, Reproduction
from .network import AdaptivePolicy
from .perception import OBSERVATION_SIZE, channel_index
from .simulation import Simulation
from .social import SOCIAL_PHYSICAL_SIZE


def mutual_information(table: list[list[int]]) -> float:
    """Discrete mutual information in bits for an analysis contingency table."""
    total = sum(sum(row) for row in table)
    if total == 0:
        return 0.0
    row_totals = [sum(row) for row in table]
    columns = max((len(row) for row in table), default=0)
    column_totals = [sum(row[column] for row in table) for column in range(columns)]
    result = 0.0
    for row_index, row in enumerate(table):
        for column, count in enumerate(row):
            if count:
                result += count / total * math.log2(
                    count * total / (row_totals[row_index] * column_totals[column])
                )
    return result


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

    def temporal_delta(
        animals, cue, current, indices, previous_actions, previous_outcomes,
        delay=1, source="full",
    ):
        if not animals or not simulation.memory:
            return 0.0, 0.0
        total = 0.0
        absolute_total = 0.0
        # Probes do not consume simulation RNG. A bounded cohort keeps reporting cheap.
        for animal in sorted(animals, key=lambda item: item.id)[:24]:
            saved = animal.adaptive_policy.to_dict()
            baseline = AdaptivePolicy.from_dict(saved)
            baseline.reset_runtime_memory()
            baseline.advance(current, use_memory=True)
            for _ in range(delay - 1):
                baseline.advance(current, use_memory=True)
            baseline_preferences = baseline.advance(current, use_memory=True)
            baseline_probabilities = animal.arbiter.probabilities(
                animal.instinct.preferences(current),
                baseline_preferences,
                adaptive_enabled=True,
            )
            contextual = AdaptivePolicy.from_dict(saved)
            contextual.reset_runtime_memory()
            contextual.advance(cue if source in ("perception", "full") else current, use_memory=True)
            if source in ("action", "full"):
                contextual.previous_actions = list(previous_actions)
            if source in ("outcome", "full"):
                contextual.previous_outcomes = list(previous_outcomes)
            for _ in range(delay - 1):
                contextual.advance(current, use_memory=True)
            contextual_preferences = contextual.advance(current, use_memory=True)
            contextual_probabilities = animal.arbiter.probabilities(
                animal.instinct.preferences(current),
                contextual_preferences,
                adaptive_enabled=True,
            )
            difference = sum(contextual_probabilities[index] for index in indices)
            difference -= sum(baseline_probabilities[index] for index in indices)
            total += difference
            absolute_total += abs(difference)
        count = min(24, len(animals))
        return total / count, absolute_total / count

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
    blank = probe(energy=0.5)

    def social_counterfactuals():
        if not simulation.social_memory:
            return {
                "social_good_bad_action_effect": 0.0,
                "social_familiar_unseen_action_effect": 0.0,
                "social_embedding_distance": 0.0,
            }
        good_bad=[]; familiar=[]; distances=[]
        physical=[0.0]*SOCIAL_PHYSICAL_SIZE
        physical[0]=physical[2]=0.25; physical[7]=1.0; physical[9]=0.5
        for animal in sorted(simulation.organisms.values(),key=lambda item:item.id)[:32]:
            entries=animal.adaptive_policy.social_memory
            if not entries:
                continue
            ranked=sorted(entries.items(),key=lambda item:item[1].get("outcome_trace",0.0))
            pairs=[(ranked[-1][0],ranked[0][0])] if len(ranked)>=2 else []
            known=ranked[-1][0]
            def action_probabilities(identity):
                policy=AdaptivePolicy.from_dict(animal.adaptive_policy.to_dict())
                policy.reset_runtime_memory()
                learned=policy.preferences(
                    blank,use_memory=True,
                    social_slots=[{"id":identity,"features":physical}],
                    social_enabled=True,step=simulation.step_count,
                )
                return animal.arbiter.probabilities(
                    animal.instinct.preferences(blank),learned,True
                )
            known_probabilities=action_probabilities(known)
            unseen_probabilities=action_probabilities(-1)
            familiar.append(max(abs(a-b) for a,b in zip(known_probabilities,unseen_probabilities)))
            for good,bad in pairs:
                good_probabilities=action_probabilities(good)
                bad_probabilities=action_probabilities(bad)
                good_bad.append(max(abs(a-b) for a,b in zip(good_probabilities,bad_probabilities)))
                first=entries[good]["embedding"]; second=entries[bad]["embedding"]
                distances.append(sum((a-b)**2 for a,b in zip(first,second))**.5)
        return {
            "social_good_bad_action_effect": round(sum(good_bad)/max(1,len(good_bad)),5),
            "social_familiar_unseen_action_effect": round(sum(familiar)/max(1,len(familiar)),5),
            "social_embedding_distance": round(sum(distances)/max(1,len(distances)),5),
        }

    food_memory_delta, food_memory_effect = temporal_delta(
        herbivores,
        hungry_food,
        blank,
        [Locomotion.FORWARD],
        [Locomotion.FORWARD, Effort.CRUISE, Interaction.FEED, Reproduction.DEFER],
        [0.8, 0.5, 2.0, 0.01],
    )
    escape_turn_memory_delta, escape_turn_memory_effect = temporal_delta(
        herbivores,
        danger_ahead,
        blank,
        [Locomotion.TURN_LEFT, Locomotion.TURN_RIGHT],
        [Locomotion.TURN_LEFT, Effort.SPRINT, Interaction.NONE, Reproduction.DEFER],
        [-0.1, -0.2, 0.01, 0.01],
    )
    escape_sprint_memory_delta, escape_sprint_memory_effect = temporal_delta(
        herbivores,
        danger_ahead,
        blank,
        [4 + Effort.SPRINT],
        [Locomotion.TURN_LEFT, Effort.SPRINT, Interaction.NONE, Reproduction.DEFER],
        [-0.1, -0.2, 0.01, 0.01],
    )
    pursuit_memory_delta, pursuit_memory_effect = temporal_delta(
        predators,
        prey_ahead,
        blank,
        [Locomotion.FORWARD],
        [Locomotion.FORWARD, Effort.CRUISE, Interaction.ATTACK, Reproduction.DEFER],
        [0.01, -0.05, -0.25, 0.01],
    )
    failed_pursuit_low_effort_delta, failed_pursuit_low_effort_effect = temporal_delta(
        predators,
        prey_ahead,
        blank,
        [4 + Effort.LOW],
        [Locomotion.FORWARD, Effort.SPRINT, Interaction.ATTACK, Reproduction.DEFER],
        [-0.4, -0.8, -0.4, 0.01],
    )
    temporal_probe_effects = {}
    probe_cases = {
        "food": (herbivores, hungry_food, [Locomotion.FORWARD],
                 [Locomotion.FORWARD, Effort.CRUISE, Interaction.FEED, Reproduction.DEFER],
                 [0.8, 0.5, 2.0, 0.01]),
        "escape": (herbivores, danger_ahead, [Locomotion.TURN_LEFT, Locomotion.TURN_RIGHT],
                   [Locomotion.TURN_LEFT, Effort.SPRINT, Interaction.NONE, Reproduction.DEFER],
                   [-0.1, -0.2, 0.01, 0.01]),
        "pursuit": (predators, prey_ahead, [Locomotion.FORWARD],
                    [Locomotion.FORWARD, Effort.SPRINT, Interaction.ATTACK, Reproduction.DEFER],
                    [-0.4, -0.8, -0.4, 0.01]),
    }
    for name, (animals, cue, indices, actions, outcomes) in probe_cases.items():
        for delay in (1, 2, 4, 8):
            _, effect = temporal_delta(
                animals, cue, blank, indices, actions, outcomes, delay, "full"
            )
            temporal_probe_effects[f"memory_{name}_full_effect_d{delay}"] = round(effect, 5)
        for source in ("perception", "action", "outcome"):
            _, effect = temporal_delta(
                animals, cue, blank, indices, actions, outcomes, 2, source
            )
            temporal_probe_effects[f"memory_{name}_{source}_effect_d2"] = round(effect, 5)

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
    communication_actions = simulation.metrics.signals + simulation.metrics.silences
    token_total = sum(simulation.metrics.signal_token_counts)
    context_probabilities = {}
    context_information = {}
    for name, table in simulation.metrics.token_context_counts.items():
        present = table[1]
        context_probabilities[name] = [
            round(count / max(1, sum(present)), 5) for count in present
        ]
        context_information[name] = round(mutual_information(table), 6)
    outcome_table = [
        [simulation.metrics.token_future_outcome_counts[token][outcome]
         for token in range(9)]
        for outcome in range(3)
    ]
    social_stats=[
        animal.adaptive_policy.social_statistics(simulation.step_count)
        for animal in simulation.organisms.values()
    ]
    def mean_social(name):
        return sum(item[name] for item in social_stats)/max(1,len(social_stats))
    result = {
        "seed": simulation.seed,
        "learning": simulation.learning,
        "controller_mode": simulation.controller_mode,
        "communication": simulation.communication,
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
        "locomotion_repeat_fraction": round(
            simulation.metrics.locomotion_repeats
            / max(1, simulation.metrics.temporal_action_pairs),
            4,
        ),
        "effort_repeat_fraction": round(
            simulation.metrics.effort_repeats
            / max(1, simulation.metrics.temporal_action_pairs),
            4,
        ),
        "repeated_association_fraction": round(
            simulation.metrics.repeated_social_encounters
            / max(1, simulation.metrics.social_encounters), 4
        ),
        "same_species_encounter_fraction": round(
            simulation.metrics.same_species_encounters
            / max(1, simulation.metrics.social_encounters), 4
        ),
        "following_fraction": round(
            simulation.metrics.follow_actions
            / max(1, simulation.metrics.follow_opportunities), 4
        ),
        "predator_colocation_persistence": round(
            simulation.metrics.repeated_predator_colocations
            / max(1, simulation.metrics.predator_colocations), 4
        ),
        "signal_rate": round(simulation.metrics.signals / max(1, communication_actions), 5),
        "silence_rate": round(simulation.metrics.silences / max(1, communication_actions), 5),
        "signal_token_distribution": [
            round(count / max(1, token_total), 5)
            for count in simulation.metrics.signal_token_counts
        ],
        "mean_signal_strength": round(sum(
            strength * count
            for strength, count in enumerate(simulation.metrics.signal_strength_counts)
        ) / max(1, sum(simulation.metrics.signal_strength_counts)), 5),
        "communication_energy_cost": round(
            simulation.metrics.communication_energy_cost, 5
        ),
        "communication_energy_fraction": round(
            simulation.metrics.communication_energy_cost
            / max(0.001, simulation.metrics.energy_spent_herbivore
                  + simulation.metrics.energy_spent_predator), 6
        ),
        "messages_delivered": simulation.metrics.messages_delivered,
        "sender_receiver_species": simulation.metrics.sender_receiver_species,
        "token_given_context": context_probabilities,
        "signal_context_mutual_information_bits": context_information,
        "signal_receiver_action_mutual_information_bits": round(
            mutual_information(simulation.metrics.token_receiver_actions), 6
        ),
        "signal_future_outcome_mutual_information_bits": round(
            mutual_information(outcome_table), 6
        ),
        "mean_future_reward_by_received_token": [
            round(total / max(1, count), 5)
            for total, count in zip(
                simulation.metrics.token_future_reward_sum,
                simulation.metrics.token_future_reward_count,
            )
        ],
        "mean_social_memory_entries": round(
            mean_social("entries"), 3
        ),
        "social_table_occupancy": round(mean_social("occupancy"),4),
        "known_individual_encounter_fraction": round(mean_social("known_fraction"),4),
        "social_eviction_rate": round(mean_social("eviction_rate"),4),
        "social_capacity_evictions_mean": round(mean_social("capacity_evictions"),3),
        "social_stale_evictions_mean": round(mean_social("stale_evictions"),3),
        "social_evicted_entry_lifetime": round(mean_social("evicted_lifetime"),2),
        "social_current_entry_lifetime": round(mean_social("current_lifetime"),2),
        "social_mean_encounters_per_entry": round(mean_social("mean_encounters"),2),
        "social_top3_encounter_concentration": round(mean_social("top3_concentration"),4),
        "social_mean_dyad_max_streak": round(mean_social("mean_max_streak"),2),
        "social_mean_normalized_distance": round(mean_social("mean_distance"),4),
        "mean_partner_concentration": round(
            sum(
                max((entry["encounters"] for entry in a.adaptive_policy.social_memory.values()), default=0)
                / max(1, sum(entry["encounters"] for entry in a.adaptive_policy.social_memory.values()))
                for a in simulation.organisms.values()
            ) / max(1, len(simulation.organisms)), 4
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
        "memory_food_search_delta": round(food_memory_delta, 5),
        "memory_food_search_effect": round(food_memory_effect, 5),
        "memory_escape_turn_delta": round(escape_turn_memory_delta, 5),
        "memory_escape_turn_effect": round(escape_turn_memory_effect, 5),
        "memory_escape_sprint_delta": round(escape_sprint_memory_delta, 5),
        "memory_escape_sprint_effect": round(escape_sprint_memory_effect, 5),
        "memory_pursuit_delta": round(pursuit_memory_delta, 5),
        "memory_pursuit_effect": round(pursuit_memory_effect, 5),
        "memory_failed_pursuit_low_effort_delta": round(
            failed_pursuit_low_effort_delta, 5
        ),
        "memory_failed_pursuit_low_effort_effect": round(
            failed_pursuit_low_effort_effect, 5
        ),
        "mean_memory_activity": round(
            sum(
                sum(abs(value) for value in animal.adaptive_policy.memory)
                for animal in simulation.organisms.values()
            )
            / max(1, len(simulation.organisms)),
            5,
        ),
    }
    result.update(temporal_probe_effects)
    result.update(social_counterfactuals())
    return result


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
