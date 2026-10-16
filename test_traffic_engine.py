import unittest
from traffic_engine import congestion_level, signal_recommendation


class TrafficEngineTests(unittest.TestCase):
    def test_low_congestion(self):
        self.assertEqual(congestion_level(2, 8, 18), "Low")

    def test_medium_congestion(self):
        self.assertEqual(congestion_level(8, 8, 18), "Medium")

    def test_high_congestion(self):
        self.assertEqual(congestion_level(18, 8, 18), "High")

    def test_signal_recommendation(self):
        self.assertIn("HIGH traffic", signal_recommendation(18))
        self.assertIn("LOW traffic", signal_recommendation(1))


if __name__ == "__main__":
    unittest.main()
