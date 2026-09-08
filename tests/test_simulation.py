import unittest

from ecosystem.simulation import OBSERVATION_SIZE, Simulation
from ecosystem.reporting import summary


class SimulationTests(unittest.TestCase):
    def test_organisms_own_distinct_policy_parameters(self):
        simulation = Simulation(seed=4)
        animals = list(simulation.organisms.values())
        self.assertGreater(len(animals), 2)
        self.assertEqual(len({id(animal.adaptive_policy) for animal in animals}), len(animals))
        self.assertEqual(len({id(animal.adaptive_policy.w1) for animal in animals}), len(animals))
        self.assertEqual(len({id(animal.adaptive_policy.wr) for animal in animals}), len(animals))
        self.assertEqual(len({id(animal.adaptive_policy.memory) for animal in animals}), len(animals))
        self.assertEqual(len({id(animal.instinct) for animal in animals}), len(animals))
        self.assertEqual(len({id(animal.arbiter) for animal in animals}), len(animals))

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

    def test_instinct_only_has_no_adaptive_influence_or_updates(self):
        simulation = Simulation(seed=8, learning=False)
        founders = list(simulation.organisms.values())
        weights = [[row[:] for row in animal.adaptive_policy.w1] for animal in founders]
        simulation.run(10)
        for animal, original in zip(founders, weights):
            self.assertEqual(animal.adaptive_policy.updates, 0)
            self.assertEqual(animal.adaptive_policy.w1, original)
        self.assertEqual(simulation.metrics.learning_updates, 0)
        self.assertTrue(all(not any(animal.adaptive_policy.memory) for animal in founders))

    def test_feedforward_mode_does_not_advance_memory(self):
        simulation = Simulation(seed=8, learning=True, memory=False)
        simulation.run(10)
        self.assertGreater(simulation.metrics.learning_updates, 0)
        self.assertTrue(
            all(not any(animal.adaptive_policy.memory) for animal in simulation.organisms.values())
        )

    def test_behavior_probes_respond_to_physiology_and_local_cues(self):
        result = summary(Simulation(seed=3, learning=False))
        self.assertGreater(result["hunger_food_approach_delta"], 0.2)
        self.assertGreater(result["hunger_feed_delta"], 0.0)
        self.assertGreater(result["predator_hunger_search_delta"], 0.1)
        self.assertGreater(result["prey_flee_turn"], 0.9)
        self.assertGreater(result["predator_attack"], 0.9)

    def test_temporal_probes_require_memory_and_identical_current_input(self):
        feedforward = summary(Simulation(seed=3, learning=True, memory=False))
        recurrent = summary(Simulation(seed=3, learning=True, memory=True, social_memory=False))
        self.assertEqual(feedforward["memory_food_search_effect"], 0.0)
        self.assertEqual(feedforward["memory_pursuit_effect"], 0.0)
        self.assertEqual(recurrent["memory_food_search_effect"], 0.0)
        trained = Simulation(seed=3, learning=True, memory=True, social_memory=False)
        trained.run(20)
        trained_summary = summary(trained)
        self.assertGreater(
            max(
                trained_summary["memory_food_full_effect_d1"],
                trained_summary["memory_escape_full_effect_d2"],
                trained_summary["memory_pursuit_full_effect_d4"],
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
