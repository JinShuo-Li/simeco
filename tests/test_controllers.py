import random
import unittest

from ecosystem.controllers import ActionArbiter, InstinctController
from ecosystem.actions import Effort, Interaction, Locomotion
from ecosystem.perception import OBSERVATION_SIZE, channel_index


def observation() -> list[float]:
    return [1.0, 0.5, 0.2, 0.5, 0.1, 0.0] + [0.0] * (OBSERVATION_SIZE - 6)


class ControllerTests(unittest.TestCase):
    def test_herbivore_instinct_flees_opposite_threat(self):
        probe = observation()
        probe[channel_index("predators", 1, 0)] = 1.0  # predator ahead
        preferences = InstinctController("herbivore").preferences(probe)
        locomotion = preferences[:4]
        self.assertIn(locomotion.index(max(locomotion)), (Locomotion.TURN_LEFT, Locomotion.TURN_RIGHT))
        self.assertEqual(preferences[4:7].index(max(preferences[4:7])), Effort.SPRINT)

    def test_predator_instinct_pursues_prey(self):
        probe = observation()
        probe[channel_index("herbivores", 1, 0)] = 1.0  # prey ahead
        preferences = InstinctController("predator").preferences(probe)
        self.assertEqual(preferences[:4].index(max(preferences[:4])), Locomotion.FORWARD)

    def test_interaction_heads_are_species_specific(self):
        herbivore_probe = observation()
        herbivore_probe[4] = 1.0
        herbivore = InstinctController("herbivore").preferences(herbivore_probe)
        self.assertEqual(herbivore[7:10].index(max(herbivore[7:10])), Interaction.FEED)
        predator_probe = observation()
        predator_probe[channel_index("herbivores", 0, 0)] = 1.0
        predator = InstinctController("predator").preferences(predator_probe)
        self.assertEqual(predator[7:10].index(max(predator[7:10])), Interaction.ATTACK)

    def test_instinct_only_arbiter_ignores_adaptive_logits(self):
        arbiter = ActionArbiter(exploration=0.0)
        instinct = [0.0, 5.0, 0.0, 0.0] + [0.0] * 8
        first = arbiter.probabilities(instinct, [0.0] * 12, adaptive_enabled=False)
        second = arbiter.probabilities(instinct, [100.0] + [0.0] * 11, adaptive_enabled=False)
        self.assertEqual(first, second)
        action, _ = arbiter.choose(instinct, [100.0] * 12, False, random.Random(2))
        self.assertEqual(action.locomotion, Locomotion.FORWARD)


if __name__ == "__main__":
    unittest.main()
