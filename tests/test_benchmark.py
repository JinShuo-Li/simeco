import unittest

from ecosystem.benchmark import run_memory_benchmark


class MemoryBenchmarkTests(unittest.TestCase):
    def test_delayed_terminal_reward_trains_context_dependent_choice(self):
        result = run_memory_benchmark(seed=41, episodes=800, trials=80)
        self.assertEqual(result["before"], {"1": 0.5, "2": 0.5, "4": 0.5, "8": 0.5})
        self.assertTrue(all(value >= 0.9 for value in result["after"].values()))


if __name__ == "__main__":
    unittest.main()
