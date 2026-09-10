"""Persistent batched adaptive policies for synchronous CPU/XPU execution."""

from __future__ import annotations

import math
from typing import Any, Iterable

from .actions import ADAPTIVE_HEAD_SIZES, TOTAL_ADAPTIVE_OUTPUTS
from .social import (
    ENTITY_HIDDEN_SIZE,
    ENTITY_SLOTS,
    SOCIAL_AGGREGATE_SIZE,
    SOCIAL_EMBEDDING_SIZE,
    SOCIAL_SENSOR_SIZE,
)

PARAMETER_NAMES = (
    "w1", "b1", "w2", "b2", "wz", "uz", "bz", "wr", "ur", "br",
    "wh", "uh", "bh", "wm_out", "entity_w", "entity_b",
)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


class BatchedPolicyStore:
    """Stack independent policy parameters without sharing them across animals.

    The leading dimension is an ownership dimension. Every operation is batched,
    but gradients at slot ``i`` can only update the parameters owned by slot
    ``i``. Runtime recurrent state remains on the selected device between ticks.
    """

    def __init__(
        self,
        animals: Iterable[Any],
        device: str = "cpu",
        capacity: int | None = None,
        learning_rates: dict[str, float] | None = None,
    ):
        import torch

        animals = list(animals)
        if not animals:
            raise ValueError("at least one animal is required")
        policies = [animal.adaptive_policy for animal in animals]
        first = policies[0]
        shape = (first.inputs, first.hidden, first.memory_size, first.outputs)
        if any((p.inputs, p.hidden, p.memory_size, p.outputs) != shape for p in policies):
            raise ValueError("all batched policies must have the same architecture")

        self.torch = torch
        self.device = torch.device(device)
        # Tiny per-animal matrices are faster in one batched CPU thread than
        # repeatedly entering a large host thread pool. XPU execution is unchanged.
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        self.capacity = capacity or max(128, 1 << math.ceil(math.log2(max(2, len(animals) * 2))))
        if self.capacity < len(animals):
            raise ValueError("capacity is smaller than the initial population")
        self.inputs, self.hidden, self.memory_size, self.outputs = shape
        self.head_sizes = first.head_sizes
        self.sensor_size = len(first.entity_w[0]) - SOCIAL_EMBEDDING_SIZE
        self.ids = [-1] * self.capacity
        self.id_to_slot: dict[int, int] = {}
        self.animals: dict[int, Any] = {animal.id: animal for animal in animals}
        self.retired_slots: set[int] = set()
        self.active = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        self.learning_rates = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.residual_scales = torch.ones(self.capacity, dtype=torch.float32, device=self.device)
        self.head_baselines = torch.zeros(
            self.capacity, len(self.head_sizes), dtype=torch.float32, device=self.device
        )
        self.memory = torch.zeros(
            self.capacity, self.memory_size, dtype=torch.float32, device=self.device
        )
        self.previous_actions = torch.full(
            (self.capacity, len(self.head_sizes)), -1, dtype=torch.long, device=self.device
        )
        self.previous_outcomes = torch.zeros(
            self.capacity, len(self.head_sizes), dtype=torch.float32, device=self.device
        )
        self.trajectory: list[dict[str, Any]] = []
        self._trajectory_start: dict[str, Any] | None = None
        self.tick = 0
        self.unroll = first.unroll
        self.gamma = first.gamma

        for name in PARAMETER_NAMES:
            sample = torch.as_tensor(getattr(first, name), dtype=torch.float32)
            storage = torch.zeros((self.capacity,) + tuple(sample.shape), dtype=torch.float32)
            for slot, policy in enumerate(policies):
                storage[slot].copy_(torch.as_tensor(getattr(policy, name), dtype=torch.float32))
            setattr(self, name, storage.to(self.device).requires_grad_(True))

        for slot, animal in enumerate(animals):
            policy = animal.adaptive_policy
            self.ids[slot] = animal.id
            self.id_to_slot[animal.id] = slot
            self.active[slot] = True
            rate = (learning_rates or {}).get(animal.species, 0.03)
            self.learning_rates[slot] = rate
            self.residual_scales[slot] = policy.residual_scale
            self.head_baselines[slot] = torch.tensor(policy.head_baselines, device=self.device)
            self.memory[slot] = torch.tensor(policy.memory, device=self.device)
            if policy.previous_actions is not None:
                self.previous_actions[slot] = torch.tensor(policy.previous_actions, device=self.device)
            self.previous_outcomes[slot] = torch.tensor(policy.previous_outcomes, device=self.device)

    def prepare_entities(self, animals, slots_by_id, step, enabled=True, embeddings_enabled=True):
        """Prepare sensor/private-memory tensors without exposing raw IDs to the network."""
        sensors = []
        embeddings = []
        masks = []
        references = []
        for animal in animals:
            policy = animal.adaptive_policy
            if enabled:
                stale = [
                    identity for identity, entry in policy.social_memory.items()
                    if step - entry["last_seen"] > policy.social_stale_ticks
                ]
                for identity in stale:
                    policy._evict_social_entry(identity, step, "stale")
            animal_sensors = []
            animal_embeddings = []
            animal_references = []
            for slot in (slots_by_id.get(animal.id, []) if enabled else [])[:ENTITY_SLOTS]:
                identity = int(slot["id"])
                entry = policy.social_memory.get(identity)
                known = entry is not None
                if entry is None:
                    entry = {
                        "embedding": [0.0] * SOCIAL_EMBEDDING_SIZE,
                        "encounters": 0, "created_at": step, "last_seen": step,
                        "consecutive_encounters": 0, "max_streak": 0,
                        "distance_sum": 0.0, "outcome_trace": 0.0,
                    }
                    policy.social_memory[identity] = entry
                    policy.social_entries_created += 1
                policy.social_total_encounters += 1
                policy.social_known_encounters += int(known)
                entry["consecutive_encounters"] = (
                    entry.get("consecutive_encounters", 0) + 1
                    if known and entry["last_seen"] == step - 1 else 1
                )
                entry["max_streak"] = max(
                    entry.get("max_streak", 0), entry["consecutive_encounters"]
                )
                entry["encounters"] += 1
                entry["last_seen"] = step
                entry["distance_sum"] = entry.get("distance_sum", 0.0) + slot["features"][2]
                features = list(slot["features"][:self.sensor_size])
                features += [0.0] * (self.sensor_size - len(features))
                animal_sensors.append(features)
                animal_embeddings.append(
                    list(entry["embedding"]) if embeddings_enabled
                    else [0.0] * SOCIAL_EMBEDDING_SIZE
                )
                animal_references.append((policy, identity))
            if enabled and len(policy.social_memory) > policy.max_social_entries:
                victims = sorted(
                    policy.social_memory.items(),
                    key=lambda item: (item[1]["last_seen"], item[1]["encounters"]),
                )[:len(policy.social_memory) - policy.max_social_entries]
                for identity, _ in victims:
                    policy._evict_social_entry(identity, step, "capacity")
            count = len(animal_sensors)
            animal_sensors += [[0.0] * self.sensor_size for _ in range(ENTITY_SLOTS - count)]
            animal_embeddings += [
                [0.0] * SOCIAL_EMBEDDING_SIZE for _ in range(ENTITY_SLOTS - count)
            ]
            animal_references += [None] * (ENTITY_SLOTS - count)
            sensors.append(animal_sensors)
            embeddings.append(animal_embeddings)
            masks.append([True] * count + [False] * (ENTITY_SLOTS - count))
            references.append(animal_references)
        return sensors, embeddings, masks, references

    @property
    def parameter_count(self) -> int:
        return sum(getattr(self, name).numel() for name in PARAMETER_NAMES)

    def slots_for(self, animal_ids: Iterable[int]):
        return self.torch.tensor(
            [self.id_to_slot[animal_id] for animal_id in animal_ids],
            dtype=self.torch.long,
            device=self.device,
        )

    def synchronize_animal(self, animal) -> None:
        slot = self.id_to_slot[animal.id]
        policy = animal.adaptive_policy
        for name in PARAMETER_NAMES:
            setattr(policy, name, getattr(self, name)[slot].detach().cpu().tolist())
        policy.memory = self.memory[slot].detach().cpu().tolist()
        previous = self.previous_actions[slot].detach().cpu().tolist()
        policy.previous_actions = None if all(value < 0 for value in previous) else previous
        policy.previous_outcomes = self.previous_outcomes[slot].detach().cpu().tolist()
        policy.head_baselines = self.head_baselines[slot].detach().cpu().tolist()

    def synchronize_animals(self, animals=None) -> None:
        for animal in animals or self.animals.values():
            if animal.id in self.id_to_slot:
                self.synchronize_animal(animal)

    def retire(self, animal_id: int) -> None:
        slot = self.id_to_slot.pop(animal_id)
        self.animals.pop(animal_id, None)
        self.active[slot] = False
        self.retired_slots.add(slot)

    def add(self, animal, learning_rate: float) -> None:
        unavailable = set(self.id_to_slot.values()) | self.retired_slots
        slot = next((index for index in range(self.capacity) if index not in unavailable), None)
        if slot is None:
            self.learn_trajectory()
            unavailable = set(self.id_to_slot.values())
            slot = next((index for index in range(self.capacity) if index not in unavailable), None)
        if slot is None:
            self._grow_capacity(self.capacity * 2)
            slot = len(unavailable)
        policy = animal.adaptive_policy
        with self.torch.no_grad():
            for name in PARAMETER_NAMES:
                getattr(self, name)[slot].copy_(self.torch.tensor(
                    getattr(policy, name), dtype=self.torch.float32, device=self.device
                ))
            self.memory[slot].zero_()
            self.previous_actions[slot].fill_(-1)
            self.previous_outcomes[slot].zero_()
            self.head_baselines[slot].copy_(self.torch.tensor(
                policy.head_baselines, dtype=self.torch.float32, device=self.device
            ))
            self.learning_rates[slot] = learning_rate
            self.residual_scales[slot] = policy.residual_scale
            self.active[slot] = True
        self.ids[slot] = animal.id
        self.id_to_slot[animal.id] = slot
        self.animals[animal.id] = animal

    def _grow_capacity(self, capacity: int) -> None:
        """Grow between autograd unrolls while retaining all resident state."""
        if self.trajectory:
            raise RuntimeError("policy capacity can only grow at a TBPTT boundary")
        torch = self.torch
        old_capacity = self.capacity
        with torch.no_grad():
            for name in PARAMETER_NAMES:
                old = getattr(self, name).detach()
                grown = torch.zeros(
                    (capacity,) + tuple(old.shape[1:]), dtype=old.dtype, device=self.device
                )
                grown[:old_capacity].copy_(old)
                setattr(self, name, grown.requires_grad_(True))
            def grow_state(old, fill=0):
                shape = (capacity,) + tuple(old.shape[1:])
                grown = torch.full(shape, fill, dtype=old.dtype, device=self.device)
                grown[:old_capacity].copy_(old)
                return grown
            self.active = grow_state(self.active)
            self.learning_rates = grow_state(self.learning_rates)
            self.residual_scales = grow_state(self.residual_scales, 1)
            self.head_baselines = grow_state(self.head_baselines)
            self.memory = grow_state(self.memory)
            self.previous_actions = grow_state(self.previous_actions, -1)
            self.previous_outcomes = grow_state(self.previous_outcomes)
        self.ids.extend([-1] * (capacity - old_capacity))
        self.capacity = capacity

    def _batched_linear(self, name: str, indices, values):
        weights = getattr(self, name).index_select(0, indices)
        return self.torch.bmm(weights, values.unsqueeze(2)).squeeze(2)

    def _action_context(self, previous_actions):
        result = self.torch.zeros(
            previous_actions.shape[0], sum(self.head_sizes),
            dtype=self.torch.float32, device=self.device,
        )
        offset = 0
        for head, size in enumerate(self.head_sizes):
            choices = previous_actions[:, head]
            valid = choices >= 0
            if valid.any():
                rows = self.torch.arange(len(choices), device=self.device)[valid]
                result[rows, offset + choices[valid]] = 1.0
            offset += size
        return result

    def forward(
        self,
        animal_ids: list[int],
        observations,
        entity_sensors,
        entity_embeddings,
        entity_mask,
        instinct_preferences,
        embedding_references=None,
        use_memory: bool = True,
    ) -> dict[str, Any]:
        """Run the complete V6 adaptive and action-distribution path in one batch."""
        torch = self.torch
        if not self.trajectory and self._trajectory_start is None:
            self._trajectory_start = {
                "memory": self.memory.detach().clone(),
                "previous_actions": self.previous_actions.detach().clone(),
                "previous_outcomes": self.previous_outcomes.detach().clone(),
                "head_baselines": self.head_baselines.detach().clone(),
                "tick": self.tick,
            }
        indices = self.slots_for(animal_ids)
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        sensors = torch.as_tensor(entity_sensors, dtype=torch.float32, device=self.device)
        embeddings = torch.as_tensor(
            entity_embeddings, dtype=torch.float32, device=self.device
        ).requires_grad_(True)
        mask = torch.as_tensor(entity_mask, dtype=torch.bool, device=self.device)
        instinct = torch.as_tensor(
            instinct_preferences, dtype=torch.float32, device=self.device
        )
        if sensors.ndim != 3 or sensors.shape[:2] != mask.shape:
            raise ValueError("entity sensors and masks must have [animals, entities, ...] shape")

        entity_input = torch.cat((sensors, embeddings), dim=2)
        entity_weights = self.entity_w.index_select(0, indices)
        entity_bias = self.entity_b.index_select(0, indices)
        represented = torch.tanh(
            torch.einsum("nei,nhi->neh", entity_input, entity_weights)
            + entity_bias.unsqueeze(1)
        )
        mask_values = mask.unsqueeze(2)
        counts = mask.sum(1, keepdim=True).clamp(min=1).to(torch.float32)
        mean = (represented * mask_values).sum(1) / counts
        maximum = represented.masked_fill(~mask_values, -torch.inf).amax(1)
        maximum = torch.where(mask.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        adaptive_input = torch.cat((observations, mean, maximum), dim=1)

        encoded = torch.tanh(
            self._batched_linear("w1", indices, adaptive_input)
            + self.b1.index_select(0, indices)
        )
        old_memory = self.memory.index_select(0, indices)
        if not use_memory:
            old_memory = torch.zeros_like(old_memory)
        previous_actions = self.previous_actions.index_select(0, indices)
        previous_outcomes = self.previous_outcomes.index_select(0, indices)
        recurrent_context = torch.cat((
            encoded,
            self._action_context(previous_actions) if use_memory
            else torch.zeros(
                len(indices), sum(self.head_sizes), device=self.device
            ),
            torch.tanh(previous_outcomes) if use_memory else torch.zeros_like(previous_outcomes),
        ), dim=1)

        def gate(input_name, recurrent_name, bias_name, activation):
            return activation(
                self._batched_linear(input_name, indices, recurrent_context)
                + self._batched_linear(recurrent_name, indices, old_memory)
                + getattr(self, bias_name).index_select(0, indices)
            )

        if use_memory:
            update = gate("wz", "uz", "bz", torch.sigmoid)
            reset = gate("wr", "ur", "br", torch.sigmoid)
            candidate = torch.tanh(
                self._batched_linear("wh", indices, recurrent_context)
                + self._batched_linear("uh", indices, reset * old_memory)
                + self.bh.index_select(0, indices)
            )
            new_memory = (1.0 - update) * old_memory + update * candidate
        else:
            new_memory = torch.zeros_like(old_memory)
        self.memory = self.memory.index_copy(0, indices, new_memory)

        raw = (
            self._batched_linear("w2", indices, encoded)
            + self._batched_linear("wm_out", indices, new_memory)
            + self.b2.index_select(0, indices)
        )
        preferences = torch.tanh(raw) * self.residual_scales.index_select(0, indices).unsqueeze(1)
        probabilities = []
        offset = 0
        for head, size in enumerate(self.head_sizes):
            adaptive = preferences[:, offset:offset + size]
            if head < 4:
                logits = (instinct[:, offset:offset + size] + 0.12 * adaptive) / 0.85
                head_probability = torch.softmax(logits, dim=1)
                head_probability = 0.965 * head_probability + 0.035 / size
            else:
                head_probability = torch.softmax(adaptive / 0.35, dim=1)
            probabilities.append(head_probability)
            offset += size
        return {
            "ids": animal_ids[:], "indices": indices, "probabilities": probabilities,
            "preferences": preferences, "embeddings": embeddings, "entity_mask": mask,
            "embedding_references": embedding_references,
            "observations": observations.detach(), "entity_sensors": sensors.detach(),
            "entity_embedding_values": embeddings.detach(),
            "instinct_preferences": instinct.detach(), "use_memory": use_memory,
        }

    def record_actions(self, transition: dict[str, Any], actions) -> None:
        actions = self.torch.as_tensor(actions, dtype=self.torch.long, device=self.device)
        transition["actions"] = actions
        transition["log_probabilities"] = self.torch.stack([
            probabilities.gather(1, actions[:, head:head + 1]).squeeze(1).clamp_min(1e-12).log()
            for head, probabilities in enumerate(transition["probabilities"])
        ], dim=1)

    def finish_tick(self, transition: dict[str, Any], outcomes, terminal=None) -> bool:
        if "log_probabilities" not in transition:
            raise RuntimeError("record_actions must be called before finish_tick")
        torch = self.torch
        indices = transition["indices"]
        outcomes = torch.as_tensor(outcomes, dtype=torch.float32, device=self.device)
        terminal = torch.zeros(len(indices), dtype=torch.bool, device=self.device) if terminal is None else torch.as_tensor(
            terminal, dtype=torch.bool, device=self.device
        )
        transition["outcomes"] = outcomes
        transition["terminal"] = terminal
        self.trajectory.append(transition)
        self.previous_actions = self.previous_actions.index_copy(
            0, indices, transition["actions"]
        )
        self.previous_outcomes = self.previous_outcomes.index_copy(0, indices, outcomes)
        current_baselines = self.head_baselines.index_select(0, indices)
        self.head_baselines = self.head_baselines.index_copy(
            0, indices, 0.96 * current_baselines + 0.04 * outcomes
        ).detach()
        self.tick += 1
        if len(self.trajectory) >= self.unroll:
            self.learn_trajectory()
            return True
        return False

    def learn_trajectory(self) -> None:
        if not self.trajectory:
            return
        torch = self.torch
        future = torch.zeros(
            self.capacity, len(self.head_sizes), dtype=torch.float32, device=self.device
        )
        losses = []
        for transition in reversed(self.trajectory):
            indices = transition["indices"]
            carried = future.index_select(0, indices)
            carried = carried * (~transition["terminal"]).unsqueeze(1)
            returns = transition["outcomes"] + self.gamma * carried
            future = future.index_copy(0, indices, returns)
            advantage = (returns - self.head_baselines.index_select(0, indices)).clamp(-4.0, 4.0)
            advantage = torch.where(advantage < 0.0, advantage * 1.8, advantage)
            losses.append(-(transition["log_probabilities"] * advantage.detach()).sum())
        loss = torch.stack(losses).sum() / max(1, sum(len(t["ids"]) for t in self.trajectory))
        loss.backward()
        for transition in self.trajectory:
            gradient = transition["embeddings"].grad
            references = transition.get("embedding_references")
            if gradient is None or references is None:
                continue
            gradient = gradient.detach().cpu().tolist()
            for row, row_references in zip(gradient, references):
                for values, reference in zip(row, row_references):
                    if reference is None:
                        continue
                    policy, identity = reference
                    entry = policy.social_memory.get(identity)
                    if entry is None:
                        continue
                    entry["embedding"] = [
                        max(-2.0, min(2.0, value - 0.03 * delta))
                        for value, delta in zip(entry["embedding"], values)
                    ]
        with torch.no_grad():
            rates = self.learning_rates
            for name in PARAMETER_NAMES:
                parameter = getattr(self, name)
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.clamp(-3.0, 3.0)
                broadcast = rates.reshape((self.capacity,) + (1,) * (parameter.ndim - 1))
                parameter.add_(-broadcast * gradient)
                parameter.grad = None
        self.memory = self.memory.detach()
        self.previous_actions = self.previous_actions.detach()
        self.previous_outcomes = self.previous_outcomes.detach()
        self.trajectory.clear()
        self._trajectory_start = None
        self.retired_slots.clear()

    def snapshot_state(self) -> dict[str, Any]:
        def values(tensor):
            return tensor.detach().cpu().tolist()
        if not self.trajectory:
            return {"trajectory": [], "tick": self.tick}
        start = self._trajectory_start
        return {
            "tick": self.tick,
            "start": {
                "memory": values(start["memory"]),
                "previous_actions": values(start["previous_actions"]),
                "previous_outcomes": values(start["previous_outcomes"]),
                "head_baselines": values(start["head_baselines"]),
                "tick": start["tick"],
            },
            "trajectory": [{
                "ids": transition["ids"],
                "observations": values(transition["observations"]),
                "entity_sensors": values(transition["entity_sensors"]),
                "entity_embeddings": values(transition["entity_embedding_values"]),
                "entity_mask": values(transition["entity_mask"]),
                "instinct_preferences": values(transition["instinct_preferences"]),
                "actions": values(transition["actions"]),
                "outcomes": values(transition["outcomes"]),
                "terminal": values(transition["terminal"]),
                "embedding_identities": [
                    [None if reference is None else reference[1] for reference in row]
                    for row in (transition.get("embedding_references") or [])
                ],
                "use_memory": transition["use_memory"],
            } for transition in self.trajectory],
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        torch = self.torch
        transitions = state.get("trajectory", [])
        self.tick = state.get("tick", 0)
        if not transitions:
            return
        start = state["start"]
        self.memory = torch.tensor(start["memory"], dtype=torch.float32, device=self.device)
        self.previous_actions = torch.tensor(
            start["previous_actions"], dtype=torch.long, device=self.device
        )
        self.previous_outcomes = torch.tensor(
            start["previous_outcomes"], dtype=torch.float32, device=self.device
        )
        self.head_baselines = torch.tensor(
            start["head_baselines"], dtype=torch.float32, device=self.device
        )
        self.tick = start["tick"]
        self._trajectory_start = None
        for saved in transitions:
            references = []
            for animal_id, identities in zip(saved["ids"], saved["embedding_identities"]):
                policy = self.animals[animal_id].adaptive_policy
                references.append([
                    None if identity is None else (policy, identity) for identity in identities
                ])
            transition = self.forward(
                saved["ids"], saved["observations"], saved["entity_sensors"],
                saved["entity_embeddings"], saved["entity_mask"],
                saved["instinct_preferences"], references, saved["use_memory"],
            )
            self.record_actions(transition, saved["actions"])
            self.finish_tick(transition, saved["outcomes"], saved["terminal"])
        self.tick = state.get("tick", self.tick)

    def synchronize(self) -> None:
        if self.device.type == "xpu":
            self.torch.xpu.synchronize()
