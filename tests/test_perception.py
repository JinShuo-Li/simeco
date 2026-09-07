import unittest

from ecosystem.perception import (
    OBSERVATION_SIZE,
    EgocentricPerception,
    channel_index,
)
from ecosystem.simulation import Simulation


class PerceptionTests(unittest.TestCase):
    def test_patch_rotates_with_heading(self):
        simulation = Simulation(seed=2)
        animal = next(iter(simulation.organisms.values()))
        simulation.resources = [
            [0.0 for _ in range(simulation.config.width)]
            for _ in range(simulation.config.height)
        ]
        simulation.resources[(animal.y - 1) % simulation.config.height][animal.x] = 10.0
        cfg = simulation.config.herbivore
        animal.heading = 0
        north_facing = EgocentricPerception.encode(
            simulation.config, cfg, simulation.resources, animal, [], []
        )
        animal.heading = 1
        east_facing = EgocentricPerception.encode(
            simulation.config, cfg, simulation.resources, animal, [], []
        )
        self.assertEqual(len(north_facing), OBSERVATION_SIZE)
        self.assertEqual(north_facing[channel_index("plants", 1, 0)], 1.0)
        self.assertEqual(east_facing[channel_index("plants", 0, -1)], 1.0)

    def test_physiology_contains_hunger(self):
        simulation = Simulation(seed=4)
        animal = next(iter(simulation.organisms.values()))
        cfg = simulation.config.herbivore
        animal.energy = cfg.max_energy * 0.25
        observation = EgocentricPerception.encode(
            simulation.config, cfg, simulation.resources, animal, [], []
        )
        self.assertAlmostEqual(observation[1], 0.25)
        self.assertAlmostEqual(observation[3], 0.75)


if __name__ == "__main__":
    unittest.main()
