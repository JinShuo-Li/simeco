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


if __name__ == "__main__":
    unittest.main()
