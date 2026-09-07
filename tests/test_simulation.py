import unittest

from ecosystem.simulation import OBSERVATION_SIZE, Simulation


class SimulationTests(unittest.TestCase):
    def test_organisms_own_distinct_policy_parameters(self):
        simulation = Simulation(seed=4)
        animals = list(simulation.organisms.values())
        self.assertGreater(len(animals), 2)
        self.assertEqual(len({id(animal.policy) for animal in animals}), len(animals))
        self.assertEqual(len({id(animal.policy.w1) for animal in animals}), len(animals))

    def test_observation_and_ecological_events(self):
        simulation = Simulation(seed=3)
        animal = next(iter(simulation.organisms.values()))
        observation = simulation.observe(
            animal, simulation.species("herbivore"), simulation.species("predator")
        )
        self.assertEqual(len(observation), OBSERVATION_SIZE)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in observation))
        initial_resources = sum(map(sum, simulation.resources))
        simulation.run(80)
        self.assertNotEqual(initial_resources, sum(map(sum, simulation.resources)))
        self.assertGreater(simulation.metrics.births_herbivore, 0)
        self.assertGreater(simulation.metrics.learning_updates, 0)


if __name__ == "__main__":
    unittest.main()
