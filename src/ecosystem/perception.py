"""Compact egocentric sensing shared by instinct and adaptive controllers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SpeciesConfig, WorldConfig
    from .model import Organism

SELF_FEATURES = 6
PATCH_WIDTH = 3
PATCH_CELLS = PATCH_WIDTH * PATCH_WIDTH
CHANNELS = ("plants", "herbivores", "predators")
OBSERVATION_SIZE = SELF_FEATURES + PATCH_CELLS * len(CHANNELS)


def patch_index(forward: int, lateral: int) -> int:
    """Index a body-relative cell; both coordinates must be -1, 0, or 1."""
    return (forward + 1) * PATCH_WIDTH + lateral + 1


def channel_index(channel: str, forward: int, lateral: int) -> int:
    return SELF_FEATURES + CHANNELS.index(channel) * PATCH_CELLS + patch_index(forward, lateral)


def relative_offset(heading: int, forward: int, lateral: int) -> tuple[int, int]:
    """Convert body-relative forward/right coordinates to world dx/dy."""
    if heading == 0:  # north
        return lateral, -forward
    if heading == 1:  # east
        return forward, lateral
    if heading == 2:  # south
        return -lateral, forward
    if heading == 3:  # west
        return -forward, -lateral
    raise ValueError(f"invalid heading: {heading}")


def egocentric_vector(heading: int, dx: int, dy: int) -> tuple[int, int]:
    """Convert a shortest world-space vector into forward/right components."""
    if heading == 0:
        return -dy, dx
    if heading == 1:
        return dx, dy
    if heading == 2:
        return dy, -dx
    if heading == 3:
        return -dx, -dy
    raise ValueError(f"invalid heading: {heading}")


class EgocentricPerception:
    """Encode a 3x3 body-relative patch and normalized physiology."""

    @staticmethod
    def encode(
        world: "WorldConfig",
        species_config: "SpeciesConfig",
        resources: list[list[float]],
        organism: "Organism",
        herbivores: list["Organism"],
        predators: list["Organism"],
    ) -> list[float]:
        observation = [
            1.0,
            min(1.0, organism.energy / species_config.max_energy),
            min(1.0, organism.age / species_config.max_age),
            max(0.0, min(1.0, 1.0 - organism.energy / species_config.max_energy)),
            resources[organism.y][organism.x] / world.plant_capacity,
            max(0.0, min(1.0, organism.reproduction_progress)),
        ] + [0.0] * (PATCH_CELLS * len(CHANNELS))

        for forward in (-1, 0, 1):
            for lateral in (-1, 0, 1):
                dx, dy = relative_offset(organism.heading, forward, lateral)
                x = (organism.x + dx) % world.width
                y = (organism.y + dy) % world.height
                observation[channel_index("plants", forward, lateral)] = (
                    resources[y][x] / world.plant_capacity
                )

        for channel, animals in (("herbivores", herbivores), ("predators", predators)):
            for other in animals:
                if other.id == organism.id:
                    continue
                dx = (other.x - organism.x + world.width // 2) % world.width - world.width // 2
                dy = (other.y - organism.y + world.height // 2) % world.height - world.height // 2
                distance = abs(dx) + abs(dy)
                if distance > species_config.vision:
                    continue
                if distance == 0:
                    forward_band = lateral_band = 0
                    proximity = 1.0
                else:
                    forward, lateral = egocentric_vector(organism.heading, dx, dy)
                    forward_band = (forward > 0) - (forward < 0)
                    lateral_band = (lateral > 0) - (lateral < 0)
                    proximity = (species_config.vision + 1 - distance) / species_config.vision
                index = channel_index(channel, forward_band, lateral_band)
                observation[index] = min(1.0, observation[index] + proximity * 0.5)
        return observation
