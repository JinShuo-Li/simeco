"""A tiny dependency-free policy network owned by one organism."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def _matrix(rows: int, columns: int, rng: random.Random, scale: float) -> list[list[float]]:
    return [[rng.gauss(0.0, scale) for _ in range(columns)] for _ in range(rows)]


@dataclass(slots=True)
class TinyMLP:
    """One-hidden-layer stochastic policy trained with an immediate policy gradient.

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
    baseline: float = 0.0
    updates: int = 0
    reward_total: float = 0.0
    last_observation: list[float] | None = field(default=None, repr=False)
    last_action: int | None = field(default=None, repr=False)
    last_probabilities: list[float] | None = field(default=None, repr=False)

    @classmethod
    def random(cls, inputs: int, hidden: int, outputs: int, rng: random.Random) -> "TinyMLP":
        return cls(
            inputs=inputs,
            hidden=hidden,
            outputs=outputs,
            w1=_matrix(hidden, inputs, rng, math.sqrt(2.0 / inputs)),
            b1=[0.0] * hidden,
            w2=_matrix(outputs, hidden, rng, math.sqrt(1.0 / hidden)),
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
        peak = max(logits)
        exponents = [math.exp(max(-30.0, value - peak)) for value in logits]
        total = sum(exponents)
        return hidden, [value / total for value in exponents]

    def choose(self, observation: list[float], rng: random.Random, exploration: float = 0.04) -> int:
        _, probabilities = self._forward(observation)
        mixed = [(1.0 - exploration) * value + exploration / self.outputs for value in probabilities]
        pick = rng.random()
        cumulative = 0.0
        action = self.outputs - 1
        for index, probability in enumerate(mixed):
            cumulative += probability
            if pick <= cumulative:
                action = index
                break
        self.last_observation = list(observation)
        self.last_action = action
        self.last_probabilities = probabilities
        return action

    def learn(self, reward: float, learning_rate: float) -> None:
        """Reinforce the preceding action; negative surprises update more strongly."""
        if self.last_observation is None or self.last_action is None:
            return
        hidden, probabilities = self._forward(self.last_observation)
        advantage = max(-4.0, min(4.0, reward - self.baseline))
        rate = learning_rate * (1.8 if advantage < 0.0 else 1.0)
        output_delta = [(-probability) * advantage for probability in probabilities]
        output_delta[self.last_action] += advantage
        old_w2 = [row[:] for row in self.w2]

        for out_index in range(self.outputs):
            delta = max(-2.0, min(2.0, output_delta[out_index]))
            for hidden_index in range(self.hidden):
                self.w2[out_index][hidden_index] += rate * delta * hidden[hidden_index]
            self.b2[out_index] += rate * delta

        for hidden_index in range(self.hidden):
            propagated = sum(old_w2[out][hidden_index] * output_delta[out] for out in range(self.outputs))
            delta = max(-2.0, min(2.0, propagated * (1.0 - hidden[hidden_index] ** 2)))
            for input_index in range(self.inputs):
                self.w1[hidden_index][input_index] += rate * delta * self.last_observation[input_index]
            self.b1[hidden_index] += rate * delta

        self.baseline = 0.96 * self.baseline + 0.04 * reward
        self.reward_total += reward
        self.updates += 1

    def offspring(self, rng: random.Random, mutation_rate: float, mutation_scale: float) -> "TinyMLP":
        child = TinyMLP.from_dict(self.to_dict())
        child.last_observation = None
        child.last_action = None
        child.last_probabilities = None
        child.updates = 0
        child.reward_total = 0.0
        child.baseline *= 0.5
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
            "baseline": self.baseline,
            "updates": self.updates,
            "reward_total": self.reward_total,
            "last_observation": self.last_observation,
            "last_action": self.last_action,
            "last_probabilities": self.last_probabilities,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TinyMLP":
        return cls(**data)
