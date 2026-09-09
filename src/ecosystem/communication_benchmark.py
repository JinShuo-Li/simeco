"""Controlled hidden-cue signaling task for arbitrary protocol learning."""

from __future__ import annotations

import math
import random

from .actions import (
    TOTAL_ADAPTIVE_OUTPUTS,
    CommunicationAction,
    Effort,
    Interaction,
    Locomotion,
    Reproduction,
)
from .network import AdaptivePolicy
from .perception import OBSERVATION_SIZE
from .social import MESSAGE_FEATURE_SIZE, SOCIAL_PHYSICAL_SIZE

SENDER_ID = 101


def _observation(cue: int | None) -> list[float]:
    result = [0.0] * OBSERVATION_SIZE
    result[0] = 1.0
    result[1] = 0.5
    if cue is not None:
        result[6 + cue] = 1.0
    return result


def _message(token: int, strength: int, identity: int = SENDER_ID) -> list[dict]:
    if token == 0:
        return []
    message = [1.0]
    message += [float(index == token) for index in range(9)]
    message += [strength / 2.0, 0.25, 0.0, 0.25, 1.0 / 3.0]
    assert len(message) == MESSAGE_FEATURE_SIZE
    return [{
        "id": identity,
        "kind": "message",
        "token": token,
        "features": [0.0] * SOCIAL_PHYSICAL_SIZE + message,
    }]


def _two_choice(preferences: list[float], rng: random.Random, train: bool):
    left, right = preferences[Locomotion.TURN_LEFT], preferences[Locomotion.TURN_RIGHT]
    peak = max(left, right)
    values = [math.exp(left - peak), math.exp(right - peak)]
    left_probability = values[0] / sum(values)
    if train:
        choice = Locomotion.TURN_LEFT if rng.random() < left_probability else Locomotion.TURN_RIGHT
    else:
        choice = Locomotion.TURN_LEFT if left_probability >= 0.5 else Locomotion.TURN_RIGHT
    probabilities = [0.0, 0.0, left_probability, 1.0 - left_probability]
    return choice, probabilities


def _fixed_physical_probabilities() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0] + [1.0, 0.0, 0.0] + [1.0, 0.0, 0.0] + [1.0, 0.0]


def _benchmark_signal(preferences, rng, train):
    """Use eight arbitrary non-silence symbols; none is tied to a cue."""
    logits = preferences[13:21]
    peak = max(logits)
    values = [math.exp((value - peak) / 0.35) for value in logits]
    probabilities_nonzero = [value / sum(values) for value in values]
    if train:
        pick = rng.random()
        cumulative = 0.0
        token = 8
        for index, probability in enumerate(probabilities_nonzero, start=1):
            cumulative += probability
            if pick <= cumulative:
                token = index
                break
    else:
        token = 1 + max(range(8), key=lambda index: probabilities_nonzero[index])
    probabilities = [0.0] + probabilities_nonzero
    probabilities += [1.0, 0.0, 0.0]
    return CommunicationAction(token, 0), probabilities


def _episode(
    sender, receiver, cue, rng, train, sender_rate=0.04, receiver_rate=0.12,
    train_sender=True,
):
    sender.reset_runtime_memory()
    sender_observation = _observation(cue)
    sender_preferences = sender.advance(sender_observation, use_memory=True)
    signal, signal_probabilities = _benchmark_signal(sender_preferences, rng, train)
    sender.record_decision(
        sender_observation,
        [Locomotion.HOLD, Effort.LOW, Interaction.NONE, Reproduction.DEFER]
        + signal.indices(),
        _fixed_physical_probabilities() + signal_probabilities,
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    rewards = []
    correct = False
    # A small rollout batch lowers the variance of the sender's delayed return;
    # every receiver update still sees only its own task-success reward.
    for _ in range(4 if train else 1):
        receiver.reset_runtime_memory()
        receiver_observation = _observation(None)
        receiver_preferences = receiver.advance(
            receiver_observation,
            use_memory=True,
            social_slots=_message(signal.token, signal.strength),
            social_enabled=True,
            step=sender.updates + 1,
        )
        choice, locomotion_probabilities = _two_choice(receiver_preferences, rng, train)
        receiver.record_decision(
            receiver_observation,
            [choice, Effort.LOW, Interaction.NONE, Reproduction.DEFER, 0, 0],
            locomotion_probabilities + _fixed_physical_probabilities()[4:]
            + [1.0] + [0.0] * 8 + [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        correct = choice == (Locomotion.TURN_LEFT if cue == 0 else Locomotion.TURN_RIGHT)
        reward = 1.0 if correct else -1.0
        rewards.append(reward)
        if train:
            receiver.learn(reward, receiver_rate, [reward] + [0.0] * 5, terminal=True)
    if train and train_sender:
        sender_reward = sum(rewards) / len(rewards)
        sender.learn(
            sender_reward, sender_rate, [0.0] * 4 + [sender_reward, 0.0], terminal=True
        )
    return correct, signal.token


def _evaluate(sender, receiver, intervention: str, seed: int, trials: int = 400) -> dict:
    rng = random.Random(seed)
    correct = 0
    table = [[0] * 9 for _ in range(2)]
    actions = [[0] * 2 for _ in range(9)]
    for trial in range(trials):
        cue = trial % 2
        sender.reset_runtime_memory()
        receiver.reset_runtime_memory()
        sender_preferences = sender.advance(_observation(cue), use_memory=True)
        signal, _ = _benchmark_signal(sender_preferences, rng, False)
        token, strength = signal.token, signal.strength
        table[cue][token] += 1
        received = token
        identity = SENDER_ID
        embeddings_enabled = True
        if intervention == "blocked":
            slots = []
        else:
            if intervention == "permuted":
                received = 1 + (token % 8)
            elif intervention == "randomized":
                received = rng.randrange(1, 9) if token else 0
            elif intervention == "identity_shuffled":
                identity = 999
            elif intervention == "embeddings_disabled":
                embeddings_enabled = False
            elif intervention == "strength_randomized":
                strength = rng.randrange(3)
            slots = _message(received, strength, identity)
        preferences = receiver.advance(
            _observation(None), use_memory=True, social_slots=slots,
            social_enabled=True, social_embeddings_enabled=embeddings_enabled,
            step=sender.updates + trial + 1,
        )
        choice, _ = _two_choice(preferences, rng, False)
        actions[token][int(choice == Locomotion.TURN_RIGHT)] += 1
        correct += choice == (Locomotion.TURN_LEFT if cue == 0 else Locomotion.TURN_RIGHT)
    return {
        "accuracy": round(correct / trials, 4),
        "cue_token_counts": table,
        "token_receiver_action_counts": actions,
    }


def run_communication_benchmark(
    seed: int = 61, episodes: int = 6000, trials: int = 400
) -> dict:
    """Train two independent policies using task success as the only reward."""
    rng = random.Random(seed)
    sender = AdaptivePolicy.random(
        OBSERVATION_SIZE, 16, TOTAL_ADAPTIVE_OUTPUTS, random.Random(seed), memory_size=12
    )
    receiver = AdaptivePolicy.random(
        OBSERVATION_SIZE, 16, TOTAL_ADAPTIVE_OUTPUTS, random.Random(seed + 10000), memory_size=12
    )
    before = _evaluate(sender, receiver, "none", seed + 1, trials)
    for episode in range(episodes):
        _episode(
            sender, receiver, rng.randrange(2), rng, True,
            train_sender=episode >= min(1000, episodes // 3),
        )
    interventions = {
        name: _evaluate(sender, receiver, name, seed + 2, trials)
        for name in (
            "none", "blocked", "permuted", "randomized", "strength_randomized",
            "identity_shuffled", "embeddings_disabled",
        )
    }
    return {
        "seed": seed,
        "episodes": episodes,
        "before": before,
        "learned": interventions.pop("none"),
        "ablations": interventions,
        "reward": "task_success_only",
        "tokens_have_predefined_semantics": False,
    }
