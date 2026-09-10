"""Frozen-observation synchronous simulator with persistent batched policies."""

from __future__ import annotations

import time

from .actions import CommunicationAction, EmbodiedAction
from .batched_policy import BatchedPolicyStore
from .social import visible_individuals
from .simulation import Simulation


def _sample(probabilities, rng):
    pick = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if pick <= cumulative:
            return index
    return len(probabilities) - 1


class SynchronousSimulation(Simulation):
    """All policies observe tick t before any action changes the world.

    Environment conflicts retain the legacy simulator's seeded resolution order;
    only policy observation/inference semantics change. Signals transmitted while
    resolving a tick are therefore first observable during the following tick.
    """

    backend = "synchronous"

    def __init__(self, *args, device: str = "cpu", policy_capacity: int = 128, **kwargs):
        started = time.perf_counter()
        super().__init__(*args, **kwargs)
        if not self.learning:
            raise ValueError("the batched backend currently requires adaptive learning")
        self.device = device
        self.policy_store = BatchedPolicyStore(
            self.organisms.values(),
            device=device,
            capacity=policy_capacity,
            learning_rates={
                "herbivore": self.config.herbivore.learning_rate,
                "predator": self.config.predator.learning_rate,
            },
        )
        self.timings = {
            "observation": 0.0,
            "policy_forward": 0.0,
            "learning": 0.0,
            "environment": 0.0,
            "reporting": 0.0,
            "initialization": time.perf_counter() - started,
        }
        self._pending_transition = None

    def step(self) -> None:
        started = time.perf_counter()
        before = {name: self.timings[name] for name in (
            "observation", "policy_forward", "learning", "reporting"
        )}
        super().step()
        elapsed = time.perf_counter() - started
        measured = sum(self.timings[name] - before[name] for name in before)
        self.timings["environment"] += max(0.0, elapsed - measured)

    def _preselect_actions(self, order, herbivores, predators):
        started = time.perf_counter()
        observations = []
        instincts = []
        slots_by_id = {}
        social_by_id = {}
        messages_by_id = {}
        for animal in order:
            cfg = self.config.herbivore if animal.species == "herbivore" else self.config.predator
            observation = self.observe(animal, herbivores, predators)
            observations.append(observation)
            instincts.append(animal.instinct.preferences(observation))
            social_slots = visible_individuals(animal, order, self.config, cfg)
            message_slots = self._message_slots(animal, cfg)
            if self.social_identity_shuffle and len(social_slots) > 1:
                identities = [slot["id"] for slot in social_slots]
                social_slots = [
                    {**slot, "id": identities[(index + 1) % len(identities)]}
                    for index, slot in enumerate(social_slots)
                ]
            if self.reverse_entity_order:
                social_slots = list(reversed(social_slots))
            social_by_id[animal.id] = social_slots
            messages_by_id[animal.id] = message_slots
            slots_by_id[animal.id] = social_slots + message_slots
        sensors, embeddings, masks, references = self.policy_store.prepare_entities(
            order, slots_by_id, self.step_count, self.social_memory, self.social_embeddings
        )
        self.timings["observation"] += time.perf_counter() - started

        started = time.perf_counter()
        transition = self.policy_store.forward(
            [animal.id for animal in order], observations, sensors, embeddings, masks,
            instincts, embedding_references=references, use_memory=self.memory,
        )
        self.policy_store.synchronize()
        flat_probabilities = self.policy_store.torch.cat(
            transition["probabilities"], dim=1
        ).detach().cpu().tolist()
        preferences = transition["preferences"].detach().cpu().tolist()
        self.timings["policy_forward"] += time.perf_counter() - started

        selected = {}
        sampled = []
        for row, animal in enumerate(order):
            previous_action = (
                animal.arbiter.last_actions[:] if animal.arbiter.last_actions is not None else None
            )
            probabilities = flat_probabilities[row]
            offset = 0
            actions = []
            head_probabilities = []
            for size in self.policy_store.head_sizes:
                head = probabilities[offset:offset + size]
                actions.append(_sample(head, self.rng))
                head_probabilities.append(head)
                offset += size
            physical = EmbodiedAction.from_indices(actions[:4])
            communication = CommunicationAction(*actions[4:6]) if self.communication else CommunicationAction()
            animal.arbiter.decisions += 1
            animal.arbiter.last_probabilities = sum(head_probabilities[:4], [])
            animal.arbiter.last_instinct_preferences = instincts[row][:]
            animal.arbiter.last_adaptive_preferences = preferences[row][:]
            animal.arbiter.last_actions = actions[:4]
            selected[animal.id] = {
                "observation": observations[row], "instinct": instincts[row],
                "social_slots": social_by_id[animal.id],
                "message_slots": messages_by_id[animal.id],
                "adaptive_preferences": preferences[row], "action": physical,
                "physical_probabilities": sum(head_probabilities[:4], []),
                "communication": communication,
                "communication_probabilities": sum(head_probabilities[4:6], []),
                "previous_action": previous_action,
            }
            sampled.append(actions)
        self.policy_store.record_actions(transition, sampled)
        self._pending_transition = transition
        return selected

    def _finish_batched_step(self, selected, order, rewards, head_rewards, dead, newborns):
        started = time.perf_counter()
        outcomes = []
        terminals = []
        for animal in order:
            reward = rewards.get(animal.id, -4.0 if animal.id not in self.organisms else 0.0)
            physical = head_rewards.get(animal.id, [reward] * 4)
            outcomes.append(physical + ([reward, reward] if self.communication else []))
            terminals.append(animal.id not in self.organisms or animal.id in dead)
            animal.adaptive_policy.updates += 1
            animal.adaptive_policy.reward_total += reward
        updated = self.policy_store.finish_tick(self._pending_transition, outcomes, terminals)
        if any(terminals) and not updated:
            self.policy_store.learn_trajectory()
            updated = True
        if updated:
            for animal in order:
                animal.adaptive_policy.tbptt_updates += 1
        for animal in order:
            if animal.id not in self.organisms and animal.id in self.policy_store.id_to_slot:
                self.policy_store.retire(animal.id)
        for child in newborns:
            cfg = self.config.herbivore if child.species == "herbivore" else self.config.predator
            self.policy_store.add(child, cfg.learning_rate)
        self._pending_transition = None
        self.timings["learning"] += time.perf_counter() - started

    def _synchronize_parent_for_offspring(self, animal):
        self.policy_store.synchronize_animal(animal)

    def synchronize_policy_state(self):
        self.policy_store.synchronize_animals(self.organisms.values())

    def record_reporting_time(self, elapsed: float) -> None:
        self.timings["reporting"] += elapsed

    def _record_history(self) -> None:
        if not hasattr(self, "timings"):
            return super()._record_history()
        started = time.perf_counter()
        super()._record_history()
        self.timings["reporting"] += time.perf_counter() - started
