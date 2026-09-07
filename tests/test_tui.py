import unittest

from ecosystem.tui import sparkline


class TUITests(unittest.TestCase):
    def test_sparkline_is_bounded_and_handles_flat_data(self):
        self.assertEqual(len(sparkline(list(range(100)), 12)), 12)
        self.assertEqual(len(set(sparkline([3, 3, 3], 10))), 1)
        self.assertEqual(sparkline([], 10), "")


if __name__ == "__main__":
    unittest.main()
