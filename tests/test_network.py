import random
import unittest

from ecosystem.actions import TOTAL_ACTION_OUTPUTS, EmbodiedAction, Locomotion
from ecosystem.network import AdaptivePolicy


class AdaptivePolicyTests(unittest.TestCase):
    def test_each_offspring_has_independent_mutated_parameters(self):
        rng = random.Random(5)
        parent = AdaptivePolicy.random(4, 3, 2, rng)
        child = parent.offspring(rng, mutation_rate=1.0, mutation_scale=0.5)
        self.assertIsNot(parent.w1, child.w1)
        self.assertNotEqual(parent.w1, child.w1)
        original = parent.w1[0][0]
        child.w1[0][0] += 10
        self.assertEqual(parent.w1[0][0], original)

    def test_offspring_inherits_recurrence_but_starts_with_fresh_memory(self):
        rng = random.Random(12)
        parent = AdaptivePolicy.random(4, 3, TOTAL_ACTION_OUTPUTS, rng)
        parent.memory = [0.8] * parent.memory_size
        parent.previous_actions = [1, 2, 0, 1]
        child = parent.offspring(rng, mutation_rate=1.0, mutation_scale=0.2)
        self.assertIsNot(parent.wr, child.wr)
        self.assertNotEqual(parent.wr, child.wr)
        self.assertEqual(child.memory, [0.0] * child.memory_size)
        self.assertIsNone(child.previous_actions)
        self.assertEqual(child.previous_outcomes, [0.0] * 4)

    def test_identical_observation_depends_on_prior_cue(self):
        policy = AdaptivePolicy.random(2, 2, TOTAL_ACTION_OUTPUTS, random.Random(3), memory_size=2)
        policy.w1 = [[0.0, 2.0], [0.0, 0.0]]
        policy.wr = [[0.0] * len(policy.wr[0]) for _ in range(2)]
        policy.wr[0][0] = 2.0
        policy.wm_out[Locomotion.FORWARD][0] = 2.0
        blank = [0.0, 0.0]
        cue = [0.0, 1.0]
        policy.reset_runtime_memory()
        policy.advance(blank)
        no_history = policy.preferences(blank)[Locomotion.FORWARD]
        policy.reset_runtime_memory()
        policy.advance(cue)
        after_cue = policy.preferences(blank)[Locomotion.FORWARD]
        self.assertGreater(after_cue, no_history + 0.1)

    def test_positive_feedback_increases_chosen_action_probability(self):
        rng = random.Random(8)
        policy = AdaptivePolicy.random(3, 4, TOTAL_ACTION_OUTPUTS, rng)
        observation = [1.0, 0.2, -0.4]
        action = EmbodiedAction(1, 1, 1, 1)
        before = policy.probabilities(observation)[1]
        combined = [0.25] * 4 + [1 / 3] * 3 + [1 / 3] * 3 + [0.5] * 2
        policy.record_decision(observation, action, combined)
        policy.learn(2.0, 0.1)
        after = policy.probabilities(observation)[1]
        self.assertGreater(after, before)

    def test_head_specific_feedback_updates_only_credited_head(self):
        rng = random.Random(9)
        policy = AdaptivePolicy.random(3, 4, TOTAL_ACTION_OUTPUTS, rng)
        observation = [1.0, 0.2, -0.4]
        action = EmbodiedAction(1, 1, 1, 1)
        combined = [0.25] * 4 + [1 / 3] * 3 + [1 / 3] * 3 + [0.5] * 2
        policy.record_decision(observation, action, combined)
        before = policy.probabilities(observation)
        policy.learn(1.0, 0.1, [0.0, 0.0, 2.0, 0.0])
        after = policy.probabilities(observation)
        self.assertEqual(before[:7], after[:7])
        self.assertNotEqual(before[7:10], after[7:10])
        self.assertEqual(before[10:], after[10:])


if __name__ == "__main__":
    unittest.main()
