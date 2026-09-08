"""A cue-delay task that directly exercises recurrent temporal credit."""
from __future__ import annotations

import random

from .actions import (
    TOTAL_ACTION_OUTPUTS,
    Effort,
    EmbodiedAction,
    Interaction,
    Locomotion,
    Reproduction,
)
from .network import AdaptivePolicy
from .perception import OBSERVATION_SIZE

DELAYS = (1, 2, 4, 8)


def _observation(cue: int | None) -> list[float]:
    observation = [0.0] * OBSERVATION_SIZE
    observation[0] = 1.0
    if cue is not None:
        observation[6 + cue] = 1.0
    return observation


def _choice(policy: AdaptivePolicy, observation: list[float], rng: random.Random, train: bool):
    preferences = policy.advance(observation, use_memory=True)
    left = preferences[Locomotion.TURN_LEFT]
    right = preferences[Locomotion.TURN_RIGHT]
    peak = max(left, right)
    left_exp, right_exp = __import__("math").exp(left - peak), __import__("math").exp(right - peak)
    left_probability = left_exp / (left_exp + right_exp)
    if train:
        choice = Locomotion.TURN_LEFT if rng.random() < left_probability else Locomotion.TURN_RIGHT
    else:
        choice = Locomotion.TURN_LEFT if left_probability >= 0.5 else Locomotion.TURN_RIGHT
    probabilities = [0.0, 0.0, left_probability, 1.0 - left_probability]
    probabilities += [1.0, 0.0, 0.0] + [1.0, 0.0, 0.0] + [1.0, 0.0]
    action = EmbodiedAction(choice, Effort.LOW, Interaction.NONE, Reproduction.DEFER)
    policy.record_decision(observation, action, probabilities)
    return choice


def _episode(policy: AdaptivePolicy, delay: int, cue: int, rng: random.Random, train: bool) -> bool:
    policy.reset_runtime_memory()
    cue_observation = _observation(cue)
    _choice(policy, cue_observation, rng, train)
    if train:
        policy.learn(0.0, 0.12, [0.0] * 4)
    chosen = Locomotion.HOLD
    for blank_index in range(delay):
        chosen = _choice(policy, _observation(None), rng, train)
        correct = chosen == (Locomotion.TURN_LEFT if cue == 0 else Locomotion.TURN_RIGHT)
        if train:
            terminal = blank_index == delay - 1
            reward = (1.0 if correct else -1.0) if terminal else 0.0
            policy.learn(reward, 0.12, [reward, 0.0, 0.0, 0.0], terminal=terminal)
    return chosen == (Locomotion.TURN_LEFT if cue == 0 else Locomotion.TURN_RIGHT)


def evaluate(policy: AdaptivePolicy, seed: int, trials: int = 400) -> dict[int, float]:
    rng = random.Random(seed)
    correct = {delay: 0 for delay in DELAYS}
    per_delay = max(2, trials // len(DELAYS))
    for delay in DELAYS:
        for trial in range(per_delay):
            correct[delay] += _episode(policy, delay, trial % 2, rng, train=False)
    return {delay: correct[delay] / per_delay for delay in DELAYS}


def run_memory_benchmark(seed: int = 41, episodes: int = 6000, trials: int = 400) -> dict:
    """Train on balanced shuffled delays and return held-out deterministic accuracy."""
    rng = random.Random(seed)
    policy = AdaptivePolicy.random(
        OBSERVATION_SIZE, 16, TOTAL_ACTION_OUTPUTS, random.Random(seed), memory_size=12
    )
    # Cue plus eight blank transitions fits into a compact nine-step truncation.
    policy.unroll = 9
    before = evaluate(policy, seed + 1, trials)
    curriculum = [(delay, cue) for delay in DELAYS for cue in (0, 1)]
    for episode in range(episodes):
        delay, cue = curriculum[episode % len(curriculum)]
        _episode(policy, delay, cue, rng, train=True)
    after = evaluate(policy, seed + 2, trials)
    return {
        "seed": seed,
        "episodes": episodes,
        "architecture": [OBSERVATION_SIZE, 16, 12, TOTAL_ACTION_OUTPUTS],
        "unroll": policy.unroll,
        "gamma": policy.gamma,
        "before": {str(k): round(v, 4) for k, v in before.items()},
        "after": {str(k): round(v, 4) for k, v in after.items()},
        "tbptt_updates": policy.tbptt_updates,
    }
