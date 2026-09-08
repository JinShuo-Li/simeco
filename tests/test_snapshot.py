import gzip
import json
import tempfile
import unittest
from pathlib import Path

from ecosystem.simulation import Simulation
from ecosystem.snapshot import load_snapshot, save_snapshot


class SnapshotTests(unittest.TestCase):
    def test_snapshot_resumes_deterministically(self):
        simulation = Simulation(seed=23)
        simulation.run(12)
        with tempfile.TemporaryDirectory() as directory:
            path = save_snapshot(simulation, Path(directory) / "state.eco.gz")
            restored = load_snapshot(path)
        simulation.run(15)
        restored.run(15)
        self.assertEqual(simulation.step_count, restored.step_count)
        self.assertEqual(simulation.resources, restored.resources)
        self.assertEqual(
            [animal.to_dict() for animal in simulation.organisms.values()],
            [animal.to_dict() for animal in restored.organisms.values()],
        )
        self.assertEqual(simulation.metrics.to_dict(), restored.metrics.to_dict())

    def test_v3_snapshot_contains_embodied_controller_state(self):
        simulation = Simulation(seed=7, learning=False)
        simulation.run(2)
        with tempfile.TemporaryDirectory() as directory:
            path = save_snapshot(simulation, Path(directory) / "state.eco.gz")
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        self.assertEqual(payload["version"], 3)
        self.assertEqual(payload["controller_mode"], "instinct_only")
        animal = payload["organisms"][0]
        self.assertIn("instinct", animal)
        self.assertIn("adaptive_policy", animal)
        self.assertIn("arbiter", animal)
        self.assertIn("reproduction_progress", animal)
        self.assertIn("heading", animal)
        self.assertEqual(animal["adaptive_policy"]["inputs"], 33)
        self.assertEqual(animal["adaptive_policy"]["outputs"], 12)
        self.assertEqual(len(animal["adaptive_policy"]["head_baselines"]), 4)
        self.assertEqual(len(animal["arbiter"]["last_actions"]), 4)


if __name__ == "__main__":
    unittest.main()
