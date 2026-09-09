import random
import unittest

from ecosystem.actions import CommunicationAction
from ecosystem.communication import (
    MESSAGE_TTL,
    SIGNAL_COSTS,
    SIGNAL_RANGES,
    expire_inbox,
    message_slot,
    signal_cost,
    signal_range,
)
from ecosystem.communication_benchmark import run_communication_benchmark
from ecosystem.controllers import InstinctController
from ecosystem.simulation import Simulation


class CommunicationTests(unittest.TestCase):
    def test_instinct_only_is_silent_and_instinct_has_no_signal_heads(self):
        simulation = Simulation(seed=8, learning=False, communication=True)
        simulation.step()
        self.assertTrue(all(animal.communication.token == 0 for animal in simulation.organisms.values()))
        observation = [0.0] * 33
        self.assertEqual(len(InstinctController("herbivore").preferences(observation)), 12)
        self.assertEqual(simulation.metrics.signals, 0)

    def test_signal_range_and_energy_cost_increase_with_strength(self):
        actions = [CommunicationAction(3, strength) for strength in range(3)]
        self.assertEqual([signal_range(action) for action in actions], list(SIGNAL_RANGES))
        self.assertEqual([signal_cost(action) for action in actions], list(SIGNAL_COSTS))
        self.assertEqual(signal_cost(CommunicationAction(0, 2)), 0.0)
        self.assertEqual(signal_range(CommunicationAction(0, 2)), 0)
        self.assertEqual(sorted(SIGNAL_COSTS), list(SIGNAL_COSTS))
        self.assertEqual(sorted(SIGNAL_RANGES), list(SIGNAL_RANGES))

    def test_inbox_expires_quickly(self):
        inbox = [{"sent_step": 4}, {"sent_step": 5}]
        self.assertEqual(len(expire_inbox(inbox, 5 + MESSAGE_TTL)), 1)
        self.assertEqual(expire_inbox(inbox, 6 + MESSAGE_TTL), [])

    def test_sender_id_is_metadata_not_policy_feature(self):
        simulation = Simulation(seed=2)
        observer, sender = list(simulation.organisms.values())[:2]
        message = {
            "sender_id": 987654321, "sender_x": sender.x, "sender_y": sender.y,
            "token": 3, "strength": 1, "sent_step": 1,
        }
        slot = message_slot(observer, message, simulation.config, 4, 2)
        self.assertEqual(slot["id"], 987654321)
        self.assertNotIn(987654321, slot["features"])
        observation = [0.0] * 33
        observer.adaptive_policy.advance(
            observation, social_slots=[slot], social_enabled=True, step=2
        )
        self.assertNotIn(987654321, observer.adaptive_policy._pending["observation"])

    def test_transmission_applies_arbitrary_token_permutation(self):
        simulation = Simulation(seed=3, token_permutation=[0, 2, 1, 3, 4, 5, 6, 7, 8])
        sender, receiver = list(simulation.organisms.values())[:2]
        receiver.x, receiver.y = sender.x, sender.y
        simulation.step_count = 1
        simulation._transmit({sender.id: CommunicationAction(1, 0)})
        self.assertEqual(receiver.inbox[-1]["token"], 2)

    def test_controlled_signaling_learns_and_blocking_collapses(self):
        result = run_communication_benchmark(seed=3, episodes=4000, trials=100)
        self.assertEqual(result["before"]["accuracy"], 0.5)
        self.assertEqual(result["learned"]["accuracy"], 1.0)
        self.assertEqual(result["ablations"]["blocked"]["accuracy"], 0.5)
        self.assertLess(result["ablations"]["randomized"]["accuracy"], 0.7)


if __name__ == "__main__":
    unittest.main()
