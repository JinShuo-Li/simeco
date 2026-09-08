"""Locally visible individuals for the adaptive controller."""
from __future__ import annotations

import math

from .actions import Effort, Locomotion
from .perception import egocentric_vector

SOCIAL_SLOTS = 8
SOCIAL_PHYSICAL_SIZE = 14
SOCIAL_EMBEDDING_SIZE = 4
SOCIAL_SLOT_SIZE = SOCIAL_PHYSICAL_SIZE + SOCIAL_EMBEDDING_SIZE
SOCIAL_INPUT_SIZE = SOCIAL_SLOTS * SOCIAL_SLOT_SIZE
MAX_SOCIAL_ENTRIES = 64
SOCIAL_STALE_TICKS = 600


def _wrapped_delta(origin: int, target: int, extent: int) -> int:
    return (target - origin + extent // 2) % extent - extent // 2


def _visible_velocity(animal) -> tuple[float, float]:
    actions = animal.arbiter.last_actions
    if actions is None or actions[0] == Locomotion.HOLD:
        return 0.0, 0.0
    speed = (0.0, 0.5, 1.0)[actions[1]]
    headings = ((0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))
    dx, dy = headings[animal.heading]
    return dx * speed, dy * speed


def visible_individuals(observer, animals, world, species_config) -> list[dict]:
    """Return up to eight physical slots; IDs remain engine-side metadata."""
    visible = []
    observer_velocity = _visible_velocity(observer)
    for other in animals:
        if other.id == observer.id:
            continue
        dx = _wrapped_delta(observer.x, other.x, world.width)
        dy = _wrapped_delta(observer.y, other.y, world.height)
        distance = abs(dx) + abs(dy)
        if distance > species_config.vision:
            continue
        forward, lateral = egocentric_vector(observer.heading, dx, dy)
        other_velocity = _visible_velocity(other)
        relative_velocity = (
            other_velocity[0] - observer_velocity[0],
            other_velocity[1] - observer_velocity[1],
        )
        velocity_forward, velocity_lateral = egocentric_vector(
            observer.heading, relative_velocity[0], relative_velocity[1]
        )
        relative_heading = (other.heading - observer.heading) % 4
        angle = relative_heading * math.pi / 2.0
        other_cfg = world.herbivore if other.species == "herbivore" else world.predator
        body_condition = round(min(1.0, other.energy / other_cfg.max_energy) * 4.0) / 4.0
        action = other.arbiter.last_actions or [0, 0, 0, 0]
        features = [
            max(-1.0, min(1.0, forward / species_config.vision)),
            max(-1.0, min(1.0, lateral / species_config.vision)),
            distance / species_config.vision,
            max(-1.0, min(1.0, velocity_forward)),
            max(-1.0, min(1.0, velocity_lateral)),
            math.cos(angle),
            math.sin(angle),
            float(other.species == observer.species),
            float(other.species != observer.species),
            body_condition,
            action[0] / 3.0,
            action[1] / 2.0,
            action[2] / 2.0,
            float(action[3]),
        ]
        visible.append((distance, other.id, {"id": other.id, "features": features}))
    visible.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in visible[:SOCIAL_SLOTS]]
