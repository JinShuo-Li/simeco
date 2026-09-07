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

        positions: dict[tuple[int, int], tuple[int, int]] = {}
        for forward in (-1, 0, 1):
            for lateral in (-1, 0, 1):
                dx, dy = relative_offset(organism.heading, forward, lateral)
                x = (organism.x + dx) % world.width
                y = (organism.y + dy) % world.height
                positions[(x, y)] = (forward, lateral)
                observation[channel_index("plants", forward, lateral)] = (
                    resources[y][x] / world.plant_capacity
                )

        for channel, animals in (("herbivores", herbivores), ("predators", predators)):
            for other in animals:
                if other.id == organism.id:
                    continue
                relative = positions.get((other.x, other.y))
                if relative is None:
                    continue
                index = channel_index(channel, *relative)
                observation[index] = min(1.0, observation[index] + 1.0 / 3.0)
        return observation
