import random
import unittest

try:
    import torch
except ImportError:
    torch = None

from ecosystem.actions import TOTAL_ADAPTIVE_OUTPUTS
from ecosystem.config import WorldConfig
from ecosystem.model import Organism
from ecosystem.network import AdaptivePolicy
from ecosystem.controllers import ActionArbiter, InstinctController
from ecosystem.social import MESSAGE_FEATURE_SIZE, SOCIAL_PHYSICAL_SIZE


@unittest.skipIf(torch is None, "PyTorch is an optional accelerated-backend dependency")
class BatchedPolicyTests(unittest.TestCase):
    def animals(self, count=3):
        result = []
        for index in range(count):
            policy = AdaptivePolicy.random(
                33, 16, TOTAL_ADAPTIVE_OUTPUTS, random.Random(100 + index), memory_size=12
            )
            animal = Organism(
                id=index + 1, species="herbivore", x=0, y=0, energy=10.0,
                age=0, generation=0, parent_id=None, heading=0,
                instinct=InstinctController("herbivore"), adaptive_policy=policy,
                arbiter=ActionArbiter(),
            )
            result.append(animal)
        return result

    def test_parameter_slices_remain_independently_owned(self):
        from ecosystem.batched_policy import BatchedPolicyStore

        store = BatchedPolicyStore(self.animals(), device="cpu")
        before = store.w1.detach().clone()
        store.w1.grad = torch.zeros_like(store.w1)
        store.w1.grad[1, 0, 0] = 1.0
        with torch.no_grad():
            rates = store.learning_rates.reshape(store.capacity, 1, 1)
            store.w1.add_(-rates * store.w1.grad)
        self.assertTrue(torch.equal(store.w1[0], before[0]))
        self.assertFalse(torch.equal(store.w1[1], before[1]))
        self.assertTrue(torch.equal(store.w1[2], before[2]))

    def test_forward_and_masked_tbptt_are_batched(self):
        from ecosystem.batched_policy import BatchedPolicyStore

        animals = self.animals()
        store = BatchedPolicyStore(animals, device="cpu")
        observations = torch.linspace(-0.5, 0.5, 99).reshape(3, 33)
        sensors = torch.zeros(3, 4, SOCIAL_PHYSICAL_SIZE + MESSAGE_FEATURE_SIZE)
        embeddings = torch.zeros(3, 4, 4)
        mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)
        instincts = torch.zeros(3, 12)
        transition = store.forward(
            [animal.id for animal in animals], observations, sensors, embeddings, mask, instincts
        )
        self.assertEqual(tuple(transition["preferences"].shape), (3, 24))
        self.assertEqual([tuple(item.shape) for item in transition["probabilities"]], [
            (3, 4), (3, 3), (3, 3), (3, 2), (3, 9), (3, 3)
        ])
        actions = torch.tensor([[0, 0, 0, 0, 1, 0]] * 3)
        store.record_actions(transition, actions)
        before = store.w2.detach().clone()
        updated = False
        for _ in range(store.unroll):
            transition = store.forward(
                [animal.id for animal in animals], observations, sensors, embeddings, mask, instincts
            )
            store.record_actions(transition, actions)
            updated = store.finish_tick(transition, torch.ones(3, 6), terminal=[True] * 3)
        self.assertTrue(updated)
        self.assertFalse(torch.equal(store.w2[:3], before[:3]))
        self.assertFalse(store.trajectory)


if __name__ == "__main__":
    unittest.main()
