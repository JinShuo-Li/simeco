import unittest

from ecosystem.simulation import Simulation
from ecosystem.social import SOCIAL_PHYSICAL_SIZE, SOCIAL_SLOTS, visible_individuals
from ecosystem.social_benchmark import run_social_benchmark


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

    def test_learned_identity_association_survives_physical_match_and_ablates(self):
        result = run_social_benchmark(seed=53, episodes=200)
        self.assertEqual(result["before"]["accuracy"], 0.5)
        self.assertEqual(result["learned"]["accuracy"], 1.0)
        self.assertEqual(result["identity_shuffled"]["accuracy"], 0.0)
        self.assertEqual(result["social_memory_reset"]["accuracy"], 0.5)
        self.assertEqual(result["social_memory_disabled"]["accuracy"], 0.5)
        learned = result["learned"]["forward_probability"]
        self.assertGreater(learned["101"] - learned["202"], 0.5)


if __name__ == "__main__":
    unittest.main()
