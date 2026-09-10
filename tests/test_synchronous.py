import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from ecosystem.snapshot import load_snapshot, save_snapshot
from ecosystem.synchronous import SynchronousSimulation


@unittest.skipIf(torch is None, "PyTorch is an optional accelerated-backend dependency")
class SynchronousSimulationTests(unittest.TestCase):
    def test_all_observations_are_built_before_any_movement(self):
        simulation = SynchronousSimulation(seed=4, device="cpu")
        snapshots = []
        original = simulation.observe

        def observed(animal, herbivores, predators):
            snapshots.append(tuple((item.id, item.x, item.y) for item in simulation.organisms.values()))
            return original(animal, herbivores, predators)

        simulation.observe = observed
        simulation.step()
        initial_observations = snapshots[:64]
        self.assertEqual(len(initial_observations), 64)
        self.assertTrue(all(item == initial_observations[0] for item in initial_observations))

    def test_same_seed_repeats_exactly_on_cpu(self):
        first = SynchronousSimulation(seed=9, device="cpu")
        second = SynchronousSimulation(seed=9, device="cpu")
        first.run(9)
        second.run(9)
        first.synchronize_policy_state()
        second.synchronize_policy_state()
        self.assertEqual(first.resources, second.resources)
        self.assertEqual(
            [animal.to_dict() for animal in first.organisms.values()],
            [animal.to_dict() for animal in second.organisms.values()],
        )
        self.assertEqual(first.metrics.to_dict(), second.metrics.to_dict())

    def test_transmitted_signals_are_first_consumed_next_tick(self):
        simulation = SynchronousSimulation(seed=5, device="cpu")
        with torch.no_grad():
            # Meaning is not assigned: this only makes an arbitrary nonzero
            # token and long-range strength deterministic for the timing test.
            simulation.policy_store.b2[:, 12:21].fill_(-20.0)
            simulation.policy_store.b2[:, 15] = 20.0
            simulation.policy_store.b2[:, 21:24].fill_(-20.0)
            simulation.policy_store.b2[:, 23] = 20.0
        simulation.step()
        self.assertGreater(simulation.metrics.messages_delivered, 0)
        self.assertEqual(sum(map(sum, simulation.metrics.token_receiver_actions)), 0)
        simulation.step()
        self.assertGreater(sum(map(sum, simulation.metrics.token_receiver_actions)), 0)

    def test_snapshot_continuation_is_deterministic(self):
        original = SynchronousSimulation(seed=12, device="cpu")
        original.run(3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sync.eco.gz"
            save_snapshot(original, path)
            restored = load_snapshot(path)
            original.run(4)
            restored.run(4)
        original.synchronize_policy_state()
        restored.synchronize_policy_state()
        self.assertEqual(original.resources, restored.resources)
        self.assertEqual(
            [animal.to_dict() for animal in original.organisms.values()],
            [animal.to_dict() for animal in restored.organisms.values()],
        )
        self.assertEqual(original.metrics.to_dict(), restored.metrics.to_dict())


if __name__ == "__main__":
    unittest.main()
