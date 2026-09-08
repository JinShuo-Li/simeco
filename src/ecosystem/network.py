"""A tiny dependency-free policy network owned by one organism."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .actions import HEAD_SIZES, EmbodiedAction


def _matrix(rows: int, columns: int, rng: random.Random, scale: float) -> list[list[float]]:
    return [[rng.gauss(0.0, scale) for _ in range(columns)] for _ in range(rows)]


@dataclass(slots=True)
class AdaptivePolicy:
    """Individual learned residual policy trained with an immediate policy gradient.

    Each instance carries its parameters and reward baseline. There is deliberately
    no shared policy or replay buffer between organisms.
    """

    inputs: int
    hidden: int
    outputs: int
    w1: list[list[float]]
    b1: list[float]
    w2: list[list[float]]
    b2: list[float]
    residual_scale: float = 1.0
    baseline: float = 0.0
    head_baselines: list[float] = field(default_factory=lambda: [0.0] * len(HEAD_SIZES))
    updates: int = 0
    reward_total: float = 0.0
    last_observation: list[float] | None = field(default=None, repr=False)
    last_actions: list[int] | None = field(default=None, repr=False)
    last_probabilities: list[float] | None = field(default=None, repr=False)
    last_gradient_scale: float = field(default=1.0, repr=False)

    @classmethod
    def random(cls, inputs: int, hidden: int, outputs: int, rng: random.Random) -> "AdaptivePolicy":
        return cls(
            inputs=inputs,
            hidden=hidden,
            outputs=outputs,
            w1=_matrix(hidden, inputs, rng, math.sqrt(2.0 / inputs)),
            b1=[0.0] * hidden,
            # A neutral output layer makes learning-enabled founders begin at the
            # instinct baseline; their private hidden encoders remain distinct.
            w2=[[0.0] * hidden for _ in range(outputs)],
            b2=[0.0] * outputs,
        )

    def _forward(self, observation: list[float]) -> tuple[list[float], list[float]]:
        hidden = [
            math.tanh(sum(weight * value for weight, value in zip(row, observation)) + bias)
            for row, bias in zip(self.w1, self.b1)
        ]
        logits = [
            sum(weight * value for weight, value in zip(row, hidden)) + bias
            for row, bias in zip(self.w2, self.b2)
        ]
        return hidden, logits

    def preferences(self, observation: list[float]) -> list[float]:
        """Return bounded residual logits without changing learning state."""
        raw = self._forward(observation)[1]
        return [math.tanh(value) * self.residual_scale for value in raw]

    def record_decision(
        self,
        observation: list[float],
        action: EmbodiedAction,
        combined_probabilities: list[float],
        gradient_scale: float = 1.0,
    ) -> None:
        self.last_observation = list(observation)
        self.last_actions = action.indices()
        self.last_probabilities = combined_probabilities[:]
        self.last_gradient_scale = gradient_scale

    def probabilities(self, observation: list[float]) -> list[float]:
        """Return the adaptive component's probabilities for analysis."""
        logits = self.preferences(observation)
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
        """Reinforce each selected head from the outcome it could influence."""
        if self.last_observation is None or self.last_actions is None:
            return
        hidden, raw_preferences = self._forward(self.last_observation)
        probabilities = self.last_probabilities or self.probabilities(self.last_observation)
        outcomes = head_rewards if head_rewards is not None else [reward] * len(HEAD_SIZES)
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
            for delta, raw in zip(output_delta, raw_preferences)
        ]
        old_w2 = [row[:] for row in self.w2]

        for out_index in range(self.outputs):
            delta = max(-2.0, min(2.0, output_delta[out_index]))
            for hidden_index in range(self.hidden):
                self.w2[out_index][hidden_index] += (
                    learning_rate * delta * hidden[hidden_index]
                )
            self.b2[out_index] += learning_rate * delta

        for hidden_index in range(self.hidden):
            propagated = sum(old_w2[out][hidden_index] * output_delta[out] for out in range(self.outputs))
            delta = max(-2.0, min(2.0, propagated * (1.0 - hidden[hidden_index] ** 2)))
            for input_index in range(self.inputs):
                self.w1[hidden_index][input_index] += (
                    learning_rate * delta * self.last_observation[input_index]
                )
            self.b1[hidden_index] += learning_rate * delta

        self.baseline = 0.96 * self.baseline + 0.04 * reward
        self.reward_total += reward
        self.updates += 1

    def offspring(self, rng: random.Random, mutation_rate: float, mutation_scale: float) -> "AdaptivePolicy":
        child = AdaptivePolicy.from_dict(self.to_dict())
        child.last_observation = None
        child.last_actions = None
        child.last_probabilities = None
        child.last_gradient_scale = 1.0
        child.updates = 0
        child.reward_total = 0.0
        child.baseline *= 0.5
        child.head_baselines = [value * 0.5 for value in child.head_baselines]
        for matrix in (child.w1, child.w2):
            for row in matrix:
                for index in range(len(row)):
                    if rng.random() < mutation_rate:
                        row[index] += rng.gauss(0.0, mutation_scale)
        for vector in (child.b1, child.b2):
            for index in range(len(vector)):
                if rng.random() < mutation_rate:
                    vector[index] += rng.gauss(0.0, mutation_scale)
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
            "residual_scale": self.residual_scale,
            "baseline": self.baseline,
            "head_baselines": self.head_baselines,
            "updates": self.updates,
            "reward_total": self.reward_total,
            "last_observation": self.last_observation,
            "last_actions": self.last_actions,
            "last_probabilities": self.last_probabilities,
            "last_gradient_scale": self.last_gradient_scale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AdaptivePolicy":
        values = dict(data)
        values["w1"] = [row[:] for row in values["w1"]]
        values["b1"] = values["b1"][:]
        values["w2"] = [row[:] for row in values["w2"]]
        values["b2"] = values["b2"][:]
        values.setdefault("residual_scale", 1.0)
        values.setdefault("head_baselines", [values.get("baseline", 0.0)] * len(HEAD_SIZES))
        values.setdefault("last_gradient_scale", 1.0)
        if "last_actions" not in values:
            old_action = values.pop("last_action", None)
            values["last_actions"] = None if old_action is None else [old_action]
        if values.get("last_observation") is not None:
            values["last_observation"] = values["last_observation"][:]
        if values.get("last_probabilities") is not None:
            values["last_probabilities"] = values["last_probabilities"][:]
        if values.get("last_actions") is not None:
            values["last_actions"] = values["last_actions"][:]
        values["head_baselines"] = values["head_baselines"][:]
        return cls(**values)


# Import compatibility for analysis code written against simeco V1.
TinyMLP = AdaptivePolicy
