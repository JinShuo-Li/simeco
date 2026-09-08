import unittest

from ecosystem.simulation import Simulation
from ecosystem.social import SOCIAL_PHYSICAL_SIZE, SOCIAL_SLOTS, visible_individuals


class SocialPerceptionTests(unittest.TestCase):
    def test_visible_slots_expose_physics_without_numeric_identity_feature(self):
        simulation = Simulation(seed=2, learning=False)
        observer = next(iter(simulation.organisms.values()))
        other = list(simulation.organisms.values())[1]
        other.x, other.y = observer.x, observer.y
        slots = visible_individuals(
            observer, list(simulation.organisms.values()), simulation.config,
            simulation.config.herbivore,
        )
        self.assertLessEqual(len(slots), SOCIAL_SLOTS)
        self.assertEqual(len(slots[0]["features"]), SOCIAL_PHYSICAL_SIZE)
        self.assertEqual(slots[0]["id"], other.id)
        self.assertNotIn(other.id, slots[0]["features"])


if __name__ == "__main__":
    unittest.main()
