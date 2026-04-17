"""Tests for GetMeasurements.gini_index (mean score inequality across players)."""

import unittest

import numpy as np
import pandas as pd

from Utils.utils import GetMeasurements


def _df(player_scores: dict, player_col: str = "id_player") -> pd.DataFrame:
    rows = []
    for pid, scores in player_scores.items():
        for r, s in enumerate(scores):
            rows.append({player_col: pid, "round": r, "score": s})
    return pd.DataFrame(rows)


class TestGiniIndex(unittest.TestCase):
    def test_equal_scores_is_zero(self):
        df = _df({1: [5.0, 5.0], 2: [5.0, 5.0], 3: [5.0, 5.0]})
        g = GetMeasurements.gini_index(df)
        self.assertAlmostEqual(g, 0.0, places=9)
        self.assertGreaterEqual(g, 0.0)

    def test_max_inequality_two_players(self):
        df = _df({1: [0.0, 0.0], 2: [10.0, 10.0]})
        g = GetMeasurements.gini_index(df)
        self.assertAlmostEqual(g, 0.5, places=9)
        self.assertTrue(0.0 <= g <= 1.0)

    def test_max_inequality_many_players(self):
        n = 5
        data = {i: ([1.0] if i == 0 else [0.0] * 3) for i in range(n)}
        df = _df(data)
        g = GetMeasurements.gini_index(df)
        self.assertAlmostEqual(g, (n - 1) / n, places=9)
        self.assertGreaterEqual(g, 0.0)

    def test_single_player(self):
        df = _df({1: [3.0, 7.0]})
        g = GetMeasurements.gini_index(df)
        self.assertAlmostEqual(g, 0.0, places=9)

    def test_all_zero_returns_zero_not_nan(self):
        df = _df({1: [0.0, 0.0], 2: [0.0, 0.0]})
        g = GetMeasurements.gini_index(df)
        self.assertEqual(g, 0.0)
        self.assertTrue(np.isfinite(g))

    def test_negative_total_mean_returns_zero_not_negative(self):
        """Lorenz formula is invalid when sum of mean scores <= 0."""
        df = _df({1: [-10.0, -10.0], 2: [5.0, 5.0], 3: [5.0, 5.0]})
        g = GetMeasurements.gini_index(df)
        self.assertEqual(g, 0.0)
        self.assertGreaterEqual(g, 0.0)

    def test_accepts_player_column_name(self):
        df = _df({1: [2.0], 2: [2.0]}, player_col="player")
        g = GetMeasurements.gini_index(df)
        self.assertAlmostEqual(g, 0.0, places=9)

    def test_random_positive_scores_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            n_players = int(rng.integers(2, 8))
            data = {
                i: rng.uniform(0.0, 10.0, size=int(rng.integers(1, 5))).tolist()
                for i in range(n_players)
            }
            df = _df(data)
            g = GetMeasurements.gini_index(df)
            self.assertGreaterEqual(g, 0.0, msg=str(g))
            self.assertLessEqual(g, 1.0 + 1e-9, msg=str(g))
            self.assertTrue(np.isfinite(g))


if __name__ == "__main__":
    unittest.main()
