"""A tiny dependency-free recurrent policy owned by one organism."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

from .actions import HEAD_SIZES, TOTAL_ACTION_OUTPUTS, EmbodiedAction

OUTCOME_SIZE = len(HEAD_SIZES)


def _matrix(rows: int, columns: int, rng: random.Random, scale: float) -> list[list[float]]:
    return [[rng.gauss(0.0, scale) for _ in range(columns)] for _ in range(rows)]


def _derived_rng(values: list[list[float]] | tuple) -> random.Random:
    """Make private randomness without advancing the simulation RNG."""
    digest = hashlib.sha256(repr(values).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _action_context(actions: list[int] | None) -> list[float]:
    context = [0.0] * TOTAL_ACTION_OUTPUTS
    if actions is None:
        return context
    offset = 0
    for size, action in zip(HEAD_SIZES, actions):
        if 0 <= action < size:
            context[offset + action] = 1.0
        offset += size
    return context


@dataclass(slots=True)
class AdaptivePolicy:
    """Individual recurrent residual policy with one-step truncated learning."""

    inputs: int
    hidden: int
    outputs: int
    w1: list[list[float]]
    b1: list[float]
    w2: list[list[float]]
    b2: list[float]
    memory_size: int
    wr: list[list[float]]
    br: list[float]
    wm_out: list[list[float]]
    memory_blend: float = 0.42
    residual_scale: float = 1.0
    baseline: float = 0.0
    head_baselines: list[float] = field(default_factory=lambda: [0.0] * len(HEAD_SIZES))
    updates: int = 0
    reward_total: float = 0.0
    memory: list[float] = field(default_factory=list)
    previous_actions: list[int] | None = None
    previous_outcomes: list[float] = field(default_factory=lambda: [0.0] * OUTCOME_SIZE)
    last_observation: list[float] | None = field(default=None, repr=False)
    last_actions: list[int] | None = field(default=None, repr=False)
    last_probabilities: list[float] | None = field(default=None, repr=False)
    last_gradient_scale: float = field(default=1.0, repr=False)
    _last_hidden: list[float] | None = field(default=None, repr=False)
    _last_memory_before: list[float] | None = field(default=None, repr=False)
    _last_memory_candidate: list[float] | None = field(default=None, repr=False)
    _last_recurrent_input: list[float] | None = field(default=None, repr=False)
    _last_raw_preferences: list[float] | None = field(default=None, repr=False)
    _last_used_memory: bool = field(default=True, repr=False)

    @classmethod
    def random(
        cls,
        inputs: int,
        hidden: int,
        outputs: int,
        rng: random.Random,
        memory_size: int = 6,
    ) -> "AdaptivePolicy":
        # Preserve V3 simulation-RNG consumption: only the feed-forward encoder
        # draws from the ecological RNG. Recurrent initialization is private.
        w1 = _matrix(hidden, inputs, rng, math.sqrt(2.0 / inputs))
        recurrent_rng = _derived_rng(w1)
        recurrent_inputs = hidden + memory_size + TOTAL_ACTION_OUTPUTS + OUTCOME_SIZE
        return cls(
            inputs=inputs,
            hidden=hidden,
            outputs=outputs,
            w1=w1,
            b1=[0.0] * hidden,
            w2=[[0.0] * hidden for _ in range(outputs)],
            b2=[0.0] * outputs,
            memory_size=memory_size,
            wr=_matrix(memory_size, recurrent_inputs, recurrent_rng, 0.16),
            br=[0.0] * memory_size,
            wm_out=[[0.0] * memory_size for _ in range(outputs)],
            memory=[0.0] * memory_size,
        )

    def _transition(
        self, observation: list[float], use_memory: bool
    ) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        hidden = [
            math.tanh(sum(weight * value for weight, value in zip(row, observation)) + bias)
            for row, bias in zip(self.w1, self.b1)
        ]
        memory_before = self.memory if use_memory else [0.0] * self.memory_size
        recurrent_input = (
            hidden
            + memory_before
            + (_action_context(self.previous_actions) if use_memory else [0.0] * TOTAL_ACTION_OUTPUTS)
            + (
                [math.tanh(value) for value in self.previous_outcomes]
                if use_memory
                else [0.0] * OUTCOME_SIZE
            )
        )
        if use_memory:
            candidate = [
                math.tanh(sum(weight * value for weight, value in zip(row, recurrent_input)) + bias)
                for row, bias in zip(self.wr, self.br)
            ]
            new_memory = [
                (1.0 - self.memory_blend) * old + self.memory_blend * new
                for old, new in zip(memory_before, candidate)
            ]
        else:
            candidate = [0.0] * self.memory_size
            new_memory = [0.0] * self.memory_size
        raw = [
            sum(weight * value for weight, value in zip(row, hidden))
            + sum(weight * value for weight, value in zip(memory_row, new_memory))
            + bias
            for row, memory_row, bias in zip(self.w2, self.wm_out, self.b2)
        ]
        return hidden, memory_before[:], candidate, recurrent_input, raw

    def advance(self, observation: list[float], use_memory: bool = True) -> list[float]:
        """Advance runtime memory once and return bounded residual preferences."""
        hidden, memory_before, candidate, recurrent_input, raw = self._transition(
            observation, use_memory
        )
        if use_memory:
            self.memory = [
                (1.0 - self.memory_blend) * old + self.memory_blend * new
                for old, new in zip(memory_before, candidate)
            ]
        else:
            self.memory = [0.0] * self.memory_size
        self.last_observation = list(observation)
        self._last_hidden = hidden
        self._last_memory_before = memory_before
        self._last_memory_candidate = candidate
        self._last_recurrent_input = recurrent_input
        self._last_raw_preferences = raw
        self._last_used_memory = use_memory
        return [math.tanh(value) * self.residual_scale for value in raw]

    def preferences(self, observation: list[float], use_memory: bool = True) -> list[float]:
        """Read preferences without changing recurrent runtime state."""
        raw = self._transition(observation, use_memory)[4]
        return [math.tanh(value) * self.residual_scale for value in raw]

    def record_decision(
        self,
        observation: list[float],
        action: EmbodiedAction,
        combined_probabilities: list[float],
        gradient_scale: float = 1.0,
    ) -> None:
        if self.last_observation != observation or self._last_hidden is None:
            self.advance(observation)
        self.last_actions = action.indices()
        self.last_probabilities = combined_probabilities[:]
        self.last_gradient_scale = gradient_scale

    def probabilities(self, observation: list[float], use_memory: bool = True) -> list[float]:
        logits = self.preferences(observation, use_memory)
        probabilities: list[float] = []
        offset = 0
        for size in HEAD_SIZES:
            head = logits[offset : offset + size]
            peak = max(head)
            exponents = [math.exp(max(-30.0, value - peak)) for value in head]
            total = sum(exponents)
            probabilities.extend(value / total for value in exponents)
            offset += size
        return probabilities

    def learn(
        self, reward: float, learning_rate: float, head_rewards: list[float] | None = None
    ) -> None:
        """Update action heads and the current recurrent transition."""
        if (
            self.last_actions is None
            or self._last_hidden is None
            or self._last_raw_preferences is None
        ):
            return
        outcomes = list(head_rewards if head_rewards is not None else [reward] * len(HEAD_SIZES))
        probabilities = self.last_probabilities or self.probabilities(
            self.last_observation or [0.0] * self.inputs
        )
        output_delta = [0.0] * self.outputs
        offset = 0
        for head_index, (size, action) in enumerate(zip(HEAD_SIZES, self.last_actions)):
            advantage = max(
                -4.0, min(4.0, outcomes[head_index] - self.head_baselines[head_index])
            )
            rate_scale = 1.8 if advantage < 0.0 else 1.0
            for index in range(size):
                output_delta[offset + index] = (
                    -probabilities[offset + index] * advantage * rate_scale
                )
            output_delta[offset + action] += advantage * rate_scale
            self.head_baselines[head_index] = (
                0.96 * self.head_baselines[head_index] + 0.04 * outcomes[head_index]
            )
            offset += size
        output_delta = [
            delta
            * self.last_gradient_scale
            * self.residual_scale
            * (1.0 - math.tanh(raw) ** 2)
            for delta, raw in zip(output_delta, self._last_raw_preferences)
        ]
        old_w2 = [row[:] for row in self.w2]
        old_wm_out = [row[:] for row in self.wm_out]
        memory_features = self.memory if self._last_used_memory else [0.0] * self.memory_size

        for out_index in range(self.outputs):
            delta = max(-2.0, min(2.0, output_delta[out_index]))
            for hidden_index in range(self.hidden):
                self.w2[out_index][hidden_index] += (
                    learning_rate * delta * self._last_hidden[hidden_index]
                )
            for memory_index in range(self.memory_size):
                self.wm_out[out_index][memory_index] += (
                    learning_rate * delta * memory_features[memory_index]
                )
            self.b2[out_index] += learning_rate * delta

        hidden_gradient = [
            sum(old_w2[out][index] * output_delta[out] for out in range(self.outputs))
            for index in range(self.hidden)
        ]
        if self._last_used_memory:
            memory_gradient = [
                sum(old_wm_out[out][index] * output_delta[out] for out in range(self.outputs))
                for index in range(self.memory_size)
            ]
            recurrent_delta = [
                max(
                    -2.0,
                    min(
                        2.0,
                        gradient
                        * self.memory_blend
                        * (1.0 - candidate**2),
                    ),
                )
                for gradient, candidate in zip(memory_gradient, self._last_memory_candidate or [])
            ]
            old_wr = [row[:] for row in self.wr]
            for memory_index, delta in enumerate(recurrent_delta):
                for input_index, value in enumerate(self._last_recurrent_input or []):
                    self.wr[memory_index][input_index] += learning_rate * delta * value
                self.br[memory_index] += learning_rate * delta
            for hidden_index in range(self.hidden):
                hidden_gradient[hidden_index] += sum(
                    old_wr[memory_index][hidden_index] * recurrent_delta[memory_index]
                    for memory_index in range(self.memory_size)
                )

        for hidden_index, gradient in enumerate(hidden_gradient):
            delta = max(
                -2.0,
                min(2.0, gradient * (1.0 - self._last_hidden[hidden_index] ** 2)),
            )
            for input_index in range(self.inputs):
                self.w1[hidden_index][input_index] += (
                    learning_rate * delta * (self.last_observation or [0.0] * self.inputs)[input_index]
                )
            self.b1[hidden_index] += learning_rate * delta

        self.baseline = 0.96 * self.baseline + 0.04 * reward
        self.reward_total += reward
        self.updates += 1
        self.previous_actions = self.last_actions[:]
        self.previous_outcomes = outcomes

    def reset_runtime_memory(self) -> None:
        self.memory = [0.0] * self.memory_size
        self.previous_actions = None
        self.previous_outcomes = [0.0] * OUTCOME_SIZE
        self.last_observation = None
        self.last_actions = None
        self.last_probabilities = None
        self.last_gradient_scale = 1.0
        self._last_hidden = None
        self._last_memory_before = None
        self._last_memory_candidate = None
        self._last_recurrent_input = None
        self._last_raw_preferences = None
        self._last_used_memory = True

    def offspring(
        self, rng: random.Random, mutation_rate: float, mutation_scale: float
    ) -> "AdaptivePolicy":
        child = AdaptivePolicy.from_dict(self.to_dict())
        child.updates = 0
        child.reward_total = 0.0
        child.baseline *= 0.5
        child.head_baselines = [value * 0.5 for value in child.head_baselines]
        # This loop intentionally matches V3 RNG use exactly.
        for matrix in (child.w1, child.w2):
            for row in matrix:
                for index in range(len(row)):
                    if rng.random() < mutation_rate:
                        row[index] += rng.gauss(0.0, mutation_scale)
        for vector in (child.b1, child.b2):
            for index in range(len(vector)):
                if rng.random() < mutation_rate:
                    vector[index] += rng.gauss(0.0, mutation_scale)
        # Recurrent inheritance mutates from private randomness so instinct-only
        # ecological RNG remains behaviorally equivalent to V3.
        recurrent_rng = _derived_rng(rng.getstate())
        for matrix in (child.wr, child.wm_out):
            for row in matrix:
                for index in range(len(row)):
                    if recurrent_rng.random() < mutation_rate:
                        row[index] += recurrent_rng.gauss(0.0, mutation_scale)
        for index in range(len(child.br)):
            if recurrent_rng.random() < mutation_rate:
                child.br[index] += recurrent_rng.gauss(0.0, mutation_scale)
        child.reset_runtime_memory()
        return child

    def to_dict(self) -> dict:
        return {
            "inputs": self.inputs,
            "hidden": self.hidden,
            "outputs": self.outputs,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "memory_size": self.memory_size,
            "wr": self.wr,
            "br": self.br,
            "wm_out": self.wm_out,
            "memory_blend": self.memory_blend,
            "residual_scale": self.residual_scale,
            "baseline": self.baseline,
            "head_baselines": self.head_baselines,
            "updates": self.updates,
            "reward_total": self.reward_total,
            "memory": self.memory,
            "previous_actions": self.previous_actions,
            "previous_outcomes": self.previous_outcomes,
            "last_observation": self.last_observation,
            "last_actions": self.last_actions,
            "last_probabilities": self.last_probabilities,
            "last_gradient_scale": self.last_gradient_scale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AdaptivePolicy":
        values = dict(data)
        for name in ("w1", "w2", "wr", "wm_out"):
            values[name] = [row[:] for row in values[name]]
        for name in (
            "b1",
            "b2",
            "br",
            "head_baselines",
            "memory",
            "previous_outcomes",
        ):
            values[name] = values[name][:]
        for name in ("last_observation", "last_actions", "last_probabilities", "previous_actions"):
            if values.get(name) is not None:
                values[name] = values[name][:]
        return cls(**values)


TinyMLP = AdaptivePolicy
