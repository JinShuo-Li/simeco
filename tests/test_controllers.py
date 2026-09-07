import random
import unittest

from ecosystem.controllers import ActionArbiter, InstinctController


def observation() -> list[float]:
    return [1.0, 0.5, 0.2, 0.1] + [0.0] * 12


class ControllerTests(unittest.TestCase):
    def test_herbivore_instinct_flees_opposite_threat(self):
        probe = observation()
        probe[12] = 1.0  # predator north
        preferences = InstinctController("herbivore").preferences(probe)
        self.assertEqual(preferences.index(max(preferences)), 3)  # flee south

    def test_predator_instinct_pursues_prey(self):
        probe = observation()
        probe[10] = 1.0  # prey south
        preferences = InstinctController("predator").preferences(probe)
        self.assertEqual(preferences.index(max(preferences)), 3)

    def test_instinct_only_arbiter_ignores_adaptive_logits(self):
        arbiter = ActionArbiter(exploration=0.0)
        instinct = [0.0, 5.0, 0.0, 0.0, 0.0]
        first = arbiter.probabilities(instinct, [0.0] * 5, adaptive_enabled=False)
        second = arbiter.probabilities(instinct, [100.0, 0.0, 0.0, 0.0, 0.0], adaptive_enabled=False)
        self.assertEqual(first, second)
        action, _ = arbiter.choose(instinct, [100.0] * 5, False, random.Random(2))
        self.assertEqual(action, 1)


if __name__ == "__main__":
    unittest.main()
