"""Short-lived, costly, sensor-like communication with no built-in semantics."""

from __future__ import annotations

import math
import random

from .actions import (
    COMMUNICATION_HEAD_SIZES,
    HEAD_SIZES,
    CommunicationAction,
)
from .perception import egocentric_vector
from .social import MESSAGE_FEATURE_SIZE, SOCIAL_PHYSICAL_SIZE

INBOX_CAPACITY = 6
MESSAGE_TTL = 3
SIGNAL_RANGES = (2, 4, 7)
SIGNAL_COSTS = (0.006, 0.02, 0.06)


def _softmax(values: list[float], temperature: float = 0.35) -> list[float]:
    peak = max(values)
    exponentials = [math.exp(max(-30.0, (value - peak) / temperature)) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def communication_probabilities(preferences: list[float]) -> list[float]:
    """Return probabilities for the two adaptive-only communication heads."""
    offset = sum(HEAD_SIZES)
    result: list[float] = []
    for size in COMMUNICATION_HEAD_SIZES:
        result.extend(_softmax(preferences[offset : offset + size]))
        offset += size
    return result


def choose_communication(
    preferences: list[float], rng: random.Random, enabled: bool
) -> tuple[CommunicationAction, list[float]]:
    """Sample a token and strength; a disabled adaptive layer is silent."""
    if not enabled:
        probabilities = [1.0] + [0.0] * (COMMUNICATION_HEAD_SIZES[0] - 1)
        probabilities += [1.0] + [0.0] * (COMMUNICATION_HEAD_SIZES[1] - 1)
        return CommunicationAction(), probabilities
    probabilities = communication_probabilities(preferences)
    actions: list[int] = []
    offset = 0
    for size in COMMUNICATION_HEAD_SIZES:
        pick = rng.random()
        cumulative = 0.0
        selected = size - 1
        for index in range(size):
            cumulative += probabilities[offset + index]
            if pick <= cumulative:
                selected = index
                break
        actions.append(selected)
        offset += size
    return CommunicationAction(*actions), probabilities


def signal_cost(action: CommunicationAction) -> float:
    return 0.0 if action.token == 0 else SIGNAL_COSTS[action.strength]


def signal_range(action: CommunicationAction) -> int:
    return 0 if action.token == 0 else SIGNAL_RANGES[action.strength]


def expire_inbox(inbox: list[dict], step: int) -> list[dict]:
    return [message for message in inbox if step - message["sent_step"] <= MESSAGE_TTL]


def message_slot(observer, message: dict, world, vision: int, step: int) -> dict:
    """Encode a received message; sender ID remains lookup-only metadata."""
    dx = (message["sender_x"] - observer.x + world.width // 2) % world.width - world.width // 2
    dy = (message["sender_y"] - observer.y + world.height // 2) % world.height - world.height // 2
    forward, lateral = egocentric_vector(observer.heading, dx, dy)
    distance = abs(dx) + abs(dy)
    token = int(message["token"])
    strength = int(message["strength"])
    message_features = [1.0]
    message_features += [float(index == token) for index in range(9)]
    message_features += [strength / 2.0]
    message_features += [
        max(-1.0, min(1.0, forward / max(1, vision))),
        max(-1.0, min(1.0, lateral / max(1, vision))),
        min(1.0, distance / max(1, SIGNAL_RANGES[-1])),
        min(1.0, (step - message["sent_step"]) / MESSAGE_TTL),
    ]
    assert len(message_features) == MESSAGE_FEATURE_SIZE
    return {
        "id": int(message["sender_id"]),
        "features": [0.0] * SOCIAL_PHYSICAL_SIZE + message_features,
        "kind": "message",
        "token": token,
    }
