"""Controlled individual-association benchmark with identical physical input."""
from __future__ import annotations

import math
import random

from .actions import (
    TOTAL_ACTION_OUTPUTS, Effort, EmbodiedAction, Interaction, Locomotion, Reproduction
)
from .network import AdaptivePolicy
from .perception import OBSERVATION_SIZE
from .social import SOCIAL_PHYSICAL_SIZE

IDENTITIES = (101, 202)
TARGETS = {101: Locomotion.FORWARD, 202: Locomotion.TURN_LEFT}


def _physical_slot(identity: int) -> dict:
    # Identity is metadata; both policy-visible physical vectors are identical.
    features = [0.0] * SOCIAL_PHYSICAL_SIZE
    features[0] = 0.25
    features[2] = 0.25
    features[7] = 1.0
    features[9] = 0.5
    return {"id": identity, "features": features}


def _observation() -> list[float]:
    result = [0.0] * OBSERVATION_SIZE
    result[0] = 1.0
    result[1] = result[2] = 0.5
    return result


def _decision(policy, identity, rng, train, social_enabled=True):
    policy.reset_runtime_memory()
    preferences = policy.advance(
        _observation(), use_memory=True,
        social_slots=[_physical_slot(identity)] if social_enabled else None,
        social_enabled=social_enabled, step=policy.updates + 1,
    )
    forward, left = preferences[Locomotion.FORWARD], preferences[Locomotion.TURN_LEFT]
    peak = max(forward, left)
    forward_exp, left_exp = math.exp(forward - peak), math.exp(left - peak)
    forward_probability = forward_exp / (forward_exp + left_exp)
    if train:
        action_index = (
            Locomotion.FORWARD if rng.random() < forward_probability else Locomotion.TURN_LEFT
        )
    else:
        action_index = (
            Locomotion.FORWARD if forward_probability >= 0.5 else Locomotion.TURN_LEFT
        )
    probabilities = [0.0, forward_probability, 1.0 - forward_probability, 0.0]
    probabilities += [1.0, 0.0, 0.0] + [1.0, 0.0, 0.0] + [1.0, 0.0]
    action = EmbodiedAction(action_index, Effort.LOW, Interaction.NONE, Reproduction.DEFER)
    policy.record_decision(_observation(), action, probabilities)
    return action_index, forward_probability


def _evaluate(policy, mapping=None, social_enabled=True):
    mapping = mapping or {identity: identity for identity in IDENTITIES}
    correct = 0
    probabilities = {}
    for presented in IDENTITIES:
        lookup = mapping[presented]
        action, forward_probability = _decision(
            policy, lookup, random.Random(0), False, social_enabled
        )
        correct += action == TARGETS[presented]
        probabilities[str(presented)] = round(forward_probability, 5)
    return {"accuracy": correct / 2.0, "forward_probability": probabilities}


def run_social_benchmark(seed: int = 53, episodes: int = 5000) -> dict:
    rng = random.Random(seed)
    policy = AdaptivePolicy.random(
        OBSERVATION_SIZE, 16, TOTAL_ACTION_OUTPUTS, random.Random(seed), memory_size=12
    )
    before = _evaluate(policy)
    for episode in range(episodes):
        identity = IDENTITIES[episode % 2]
        action, _ = _decision(policy, identity, rng, True)
        reward = 1.0 if action == TARGETS[identity] else -1.0
        policy.learn(reward, 0.12, [reward, 0.0, 0.0, 0.0], terminal=True)
    learned = _evaluate(policy)
    shuffled = _evaluate(policy, {101: 202, 202: 101})
    saved = {key: dict(value) for key, value in policy.social_memory.items()}
    policy.reset_social_memory()
    reset = _evaluate(policy)
    policy.social_memory = saved
    disabled = _evaluate(policy, social_enabled=False)
    return {
        "seed": seed, "episodes": episodes,
        "physical_vectors_identical": True,
        "before": before, "learned": learned,
        "identity_shuffled": shuffled,
        "social_memory_reset": reset,
        "social_memory_disabled": disabled,
        "embedding_distance": round(math.sqrt(sum(
            (a-b)**2 for a,b in zip(
                policy.social_memory[101]["embedding"],
                policy.social_memory[202]["embedding"],
            )
        )), 5),
    }
