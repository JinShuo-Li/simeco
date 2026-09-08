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

    def test_v5_snapshot_contains_temporal_and_social_controller_state(self):
        simulation = Simulation(seed=7, learning=True, memory=True, social_memory=True)
        simulation.run(2)
        with tempfile.TemporaryDirectory() as directory:
            path = save_snapshot(simulation, Path(directory) / "state.eco.gz")
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        self.assertEqual(payload["version"], 7)
        self.assertEqual(payload["controller_mode"], "instinct+learning+memory+social")
        self.assertTrue(payload["memory"])
        self.assertTrue(payload["social_memory"])
        self.assertTrue(payload["social_embeddings"])
        self.assertFalse(payload["social_identity_shuffle"])
        self.assertFalse(payload["reverse_entity_order"])
        animal = payload["organisms"][0]
        self.assertIn("instinct", animal)
        self.assertIn("adaptive_policy", animal)
        self.assertIn("arbiter", animal)
        self.assertIn("reproduction_progress", animal)
        self.assertIn("heading", animal)
        self.assertEqual(animal["adaptive_policy"]["inputs"], 49)
        self.assertEqual(animal["adaptive_policy"]["base_inputs"], 33)
        self.assertEqual(animal["adaptive_policy"]["outputs"], 12)
        self.assertEqual(len(animal["adaptive_policy"]["head_baselines"]), 4)
        self.assertEqual(animal["adaptive_policy"]["hidden"], 16)
        self.assertEqual(len(animal["adaptive_policy"]["memory"]), 12)
        self.assertEqual(len(animal["adaptive_policy"]["wz"]), 12)
        self.assertEqual(len(animal["adaptive_policy"]["wr"]), 12)
        self.assertEqual(len(animal["adaptive_policy"]["wh"]), 12)
        self.assertIn("trajectory", animal["adaptive_policy"])
        self.assertIn("social_memory", animal["adaptive_policy"])
        self.assertEqual(len(animal["adaptive_policy"]["entity_w"]), 8)
        self.assertEqual(len(animal["adaptive_policy"]["entity_w"][0]), 18)
        self.assertIn("social_evictions_capacity", animal["adaptive_policy"])
        self.assertIn("social_known_encounters", animal["adaptive_policy"])
        social_entries = [
            entry
            for item in payload["organisms"]
            for entry in item["adaptive_policy"]["social_memory"].values()
        ]
        self.assertTrue(social_entries)
        self.assertTrue(all("outcome_trace" in entry for entry in social_entries))
        self.assertEqual(len(animal["adaptive_policy"]["previous_outcomes"]), 4)
        self.assertTrue(any(animal["adaptive_policy"]["memory"]))
        self.assertIsNotNone(animal["adaptive_policy"]["previous_actions"])
        self.assertEqual(len(animal["arbiter"]["last_actions"]), 4)


if __name__ == "__main__":
    unittest.main()
