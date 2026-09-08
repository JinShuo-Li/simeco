import random
import unittest

from ecosystem.actions import TOTAL_ACTION_OUTPUTS, EmbodiedAction, Locomotion
from ecosystem.network import AdaptivePolicy
from ecosystem.social import SOCIAL_PHYSICAL_SIZE


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
        policy.wz = [[0.0] * len(policy.wz[0]) for _ in range(2)]
        policy.uz = [[0.0] * 2 for _ in range(2)]
        policy.wr = [[0.0] * len(policy.wr[0]) for _ in range(2)]
        policy.ur = [[0.0] * 2 for _ in range(2)]
        policy.wh = [[0.0] * len(policy.wh[0]) for _ in range(2)]
        policy.uh = [[0.0] * 2 for _ in range(2)]
        policy.wh[0][0] = 2.0
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
        policy.learn(2.0, 0.1, terminal=True)
        after = policy.probabilities(observation)[1]
        self.assertGreater(after, before)

    def test_head_specific_feedback_updates_only_credited_head(self):
        rng = random.Random(9)
        policy = AdaptivePolicy.random(3, 4, TOTAL_ACTION_OUTPUTS, rng)
        observation = [1.0, 0.2, -0.4]
        action = EmbodiedAction(1, 1, 1, 1)
        combined = [0.25] * 4 + [1 / 3] * 3 + [1 / 3] * 3 + [0.5] * 2
        policy.advance(observation, use_memory=False)
        policy.record_decision(observation, action, combined)
        before = policy.probabilities(observation, use_memory=False)
        policy.learn(1.0, 0.1, [0.0, 0.0, 2.0, 0.0])
        after = policy.probabilities(observation, use_memory=False)
        self.assertEqual(before[:7], after[:7])
        self.assertNotEqual(before[7:10], after[7:10])
        self.assertEqual(before[10:], after[10:])

    def test_private_identity_embedding_changes_action_for_same_physics(self):
        policy = AdaptivePolicy.random(33, 16, TOTAL_ACTION_OUTPUTS, random.Random(19))
        policy.w1 = [[0.0] * policy.inputs for _ in range(policy.hidden)]
        policy.w2 = [[0.0] * policy.hidden for _ in range(policy.outputs)]
        policy.entity_w = [[0.0] * len(policy.entity_w[0]) for _ in policy.entity_w]
        policy.entity_w[0][SOCIAL_PHYSICAL_SIZE] = 2.0
        policy.w1[0][policy.base_inputs] = 2.0
        policy.w2[Locomotion.FORWARD][0] = 2.0
        policy.social_memory = {
            101: {"embedding": [1.0, 0.0, 0.0, 0.0], "encounters": 4, "last_seen": 1},
            202: {"embedding": [-1.0, 0.0, 0.0, 0.0], "encounters": 4, "last_seen": 1},
        }
        physical = [0.0] * SOCIAL_PHYSICAL_SIZE
        first = policy.preferences(
            [0.0] * 33, use_memory=False,
            social_slots=[{"id": 101, "features": physical}], social_enabled=True,
        )
        second = policy.preferences(
            [0.0] * 33, use_memory=False,
            social_slots=[{"id": 202, "features": physical}], social_enabled=True,
        )
        self.assertGreater(first[Locomotion.FORWARD], second[Locomotion.FORWARD] + 1.0)

    def test_entity_pooling_is_invariant_to_slot_order(self):
        policy = AdaptivePolicy.random(33, 16, TOTAL_ACTION_OUTPUTS, random.Random(29))
        policy.social_memory = {
            identity: {
                "embedding": [identity / 10.0, 0.1, -0.1, 0.0],
                "encounters": 2, "created_at": 0, "last_seen": 1,
                "consecutive_encounters": 1, "max_streak": 1,
                "distance_sum": 0.5, "outcome_trace": 0.0,
            }
            for identity in (1, 2, 3)
        }
        slots = [
            {"id": identity, "features": [identity / 4.0] + [0.0] * (SOCIAL_PHYSICAL_SIZE - 1)}
            for identity in (1, 2, 3)
        ]
        first = policy.preferences(
            [0.0] * 33, use_memory=False, social_slots=slots, social_enabled=True
        )
        second = policy.preferences(
            [0.0] * 33, use_memory=False, social_slots=list(reversed(slots)),
            social_enabled=True,
        )
        for before, after in zip(first, second):
            self.assertAlmostEqual(before, after, places=12)

    def test_disabling_retrieved_embeddings_removes_identity_effect(self):
        policy = AdaptivePolicy.random(33, 16, TOTAL_ACTION_OUTPUTS, random.Random(30))
        policy.social_memory = {
            identity: {
                "embedding": embedding, "encounters": 2, "created_at": 0,
                "last_seen": 1, "consecutive_encounters": 1, "max_streak": 1,
                "distance_sum": 0.5, "outcome_trace": 0.0,
            }
            for identity, embedding in ((1, [1.0, 0.0, 0.0, 0.0]), (2, [-1.0, 0.0, 0.0, 0.0]))
        }
        physical = [0.0] * SOCIAL_PHYSICAL_SIZE
        results = [
            policy.preferences(
                [0.0] * 33, use_memory=False,
                social_slots=[{"id": identity, "features": physical}],
                social_enabled=True, social_embeddings_enabled=False,
            )
            for identity in (1, 2)
        ]
        self.assertEqual(results[0], results[1])

    def test_offspring_does_not_inherit_social_memories(self):
        policy = AdaptivePolicy.random(33, 16, TOTAL_ACTION_OUTPUTS, random.Random(20))
        policy.social_memory[7] = {
            "embedding": [0.2, -0.1, 0.3, 0.0], "encounters": 9, "last_seen": 30
        }
        child = policy.offspring(random.Random(21), 0.0, 0.0)
        self.assertEqual(child.social_memory, {})

    def test_social_memory_evicts_by_capacity_and_staleness(self):
        policy = AdaptivePolicy.random(33, 16, TOTAL_ACTION_OUTPUTS, random.Random(22))
        policy.max_social_entries = 2
        policy.social_stale_ticks = 2
        physical = [0.0] * SOCIAL_PHYSICAL_SIZE
        for step, identity in enumerate((1, 2, 3), start=1):
            policy.advance(
                [0.0] * 33, social_slots=[{"id": identity, "features": physical}],
                social_enabled=True, step=step,
            )
        self.assertEqual(set(policy.social_memory), {2, 3})
        policy.advance([0.0] * 33, social_slots=[], social_enabled=True, step=6)
        self.assertEqual(policy.social_memory, {})


if __name__ == "__main__":
    unittest.main()
