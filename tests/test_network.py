import random
import unittest

from ecosystem.network import TinyMLP


class TinyMLPTests(unittest.TestCase):
    def test_each_offspring_has_independent_mutated_parameters(self):
        rng = random.Random(5)
        parent = TinyMLP.random(4, 3, 2, rng)
        child = parent.offspring(rng, mutation_rate=1.0, mutation_scale=0.5)
        self.assertIsNot(parent.w1, child.w1)
        self.assertNotEqual(parent.w1, child.w1)
        original = parent.w1[0][0]
        child.w1[0][0] += 10
        self.assertEqual(parent.w1[0][0], original)

    def test_positive_feedback_increases_chosen_action_probability(self):
        rng = random.Random(8)
        policy = TinyMLP.random(3, 4, 2, rng)
        observation = [1.0, 0.2, -0.4]
        policy.choose(observation, rng)
        action = policy.last_action
        before = policy._forward(observation)[1][action]
        policy.learn(2.0, 0.1)
        after = policy._forward(observation)[1][action]
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
