"""The embodied, multi-head action selected once per animal tick."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Locomotion(IntEnum):
    HOLD = 0
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class Effort(IntEnum):
    LOW = 0
    CRUISE = 1
    SPRINT = 2


class Interaction(IntEnum):
    NONE = 0
    FEED = 1
    ATTACK = 2


class Reproduction(IntEnum):
    DEFER = 0
    INTEND = 1


HEAD_NAMES = ("locomotion", "effort", "interaction", "reproduction")
HEAD_SIZES = (4, 3, 3, 2)
TOTAL_ACTION_OUTPUTS = sum(HEAD_SIZES)


@dataclass(frozen=True, slots=True)
class EmbodiedAction:
    locomotion: int
    effort: int
    interaction: int
    reproduction: int

    def indices(self) -> list[int]:
        return [self.locomotion, self.effort, self.interaction, self.reproduction]

    @classmethod
    def from_indices(cls, actions: list[int]) -> "EmbodiedAction":
        if len(actions) != len(HEAD_SIZES):
            raise ValueError("one action is required for each action head")
        return cls(*actions)

    def to_dict(self) -> dict[str, int]:
        return dict(zip(HEAD_NAMES, self.indices()))
