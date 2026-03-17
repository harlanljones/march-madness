"""Tests for src/features.py -- feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    build_matchup_features,
    build_tournament_matchups,
    _safe_diff,
    FEATURE_NAMES,
    N_FEATURES,
)


class TestSafeDiff:
    def test_normal(self):
        stats = pd.DataFrame({"Stat": [10.0, 5.0]}, index=[100, 200])
        assert _safe_diff(stats, 100, 200, "Stat") == pytest.approx(5.0)

    def test_missing_column(self):
        stats = pd.DataFrame({"Other": [1.0]}, index=[100])
        assert _safe_diff(stats, 100, 200, "Stat") == 0.0

    def test_missing_team(self):
        stats = pd.DataFrame({"Stat": [10.0]}, index=[100])
        assert _safe_diff(stats, 100, 999, "Stat") == 0.0

    def test_nan_value(self):
        stats = pd.DataFrame({"Stat": [10.0, np.nan]}, index=[100, 200])
        assert _safe_diff(stats, 100, 200, "Stat") == 0.0


class TestBuildMatchupFeatures:
    def test_output_shape(self, synthetic_data):
        from src.data_loader import build_season_team_stats
        stats = build_season_team_stats(synthetic_data, 2022)
        team_a, team_b = 1101, 1102
        feats = build_matchup_features(team_a, team_b, stats)
        assert feats.shape == (N_FEATURES,)
        assert feats.dtype == np.float32

    def test_feature_count_matches_names(self):
        assert N_FEATURES == len(FEATURE_NAMES)
        assert N_FEATURES == 23

    def test_antisymmetry(self, synthetic_data):
        """Features for (A, B) should negate features for (B, A) (approximately).

        The convention is lower ID = team A, so we test that flipping
        teams flips signs on the difference features.
        """
        from src.data_loader import build_season_team_stats
        stats = build_season_team_stats(synthetic_data, 2022)
        feats_ab = build_matchup_features(1101, 1102, stats)
        feats_ba = build_matchup_features(1102, 1101, stats)
        # Should be negations of each other
        np.testing.assert_allclose(feats_ab, -feats_ba, atol=1e-5)

    def test_same_team_yields_zeros(self, synthetic_data):
        """Matchup of a team against itself should produce all-zero features."""
        from src.data_loader import build_season_team_stats
        stats = build_season_team_stats(synthetic_data, 2022)
        feats = build_matchup_features(1101, 1101, stats)
        np.testing.assert_allclose(feats, 0.0, atol=1e-7)

    def test_missing_team_returns_zeros(self, synthetic_data):
        """If a team isn't in stats, features should be 0 (not crash)."""
        from src.data_loader import build_season_team_stats
        stats = build_season_team_stats(synthetic_data, 2022)
        feats = build_matchup_features(1101, 9999, stats)
        assert feats.shape == (N_FEATURES,)
        assert not np.any(np.isnan(feats))

    def test_seed_diff_direction(self, synthetic_data):
        """A 1-seed vs 4-seed should have positive SeedDiff (better seed = positive)."""
        from src.data_loader import build_season_team_stats
        stats = build_season_team_stats(synthetic_data, 2022)
        # Team 1101 is seed 1, Team 1104 is seed 4 in region W
        feats = build_matchup_features(1101, 1104, stats)
        seed_diff_idx = FEATURE_NAMES.index("SeedDiff")
        assert feats[seed_diff_idx] > 0, "1-seed should have positive SeedDiff vs 4-seed"

    def test_no_nans(self, synthetic_data):
        from src.data_loader import build_season_team_stats
        stats = build_season_team_stats(synthetic_data, 2022)
        feats = build_matchup_features(1101, 1102, stats)
        assert not np.any(np.isnan(feats))


class TestBuildTournamentMatchups:
    def test_output_shapes(self, synthetic_data):
        features, labels = build_tournament_matchups(synthetic_data, [2022])
        assert features.ndim == 2
        assert features.shape[1] == N_FEATURES
        assert labels.ndim == 1
        assert features.shape[0] == labels.shape[0]
        assert features.shape[0] > 0

    def test_labels_binary(self, synthetic_data):
        _, labels = build_tournament_matchups(synthetic_data, [2022])
        assert set(np.unique(labels)).issubset({0.0, 1.0})

    def test_no_nans(self, synthetic_data):
        features, labels = build_tournament_matchups(synthetic_data, [2022])
        assert not np.any(np.isnan(features))
        assert not np.any(np.isnan(labels))

    def test_multiple_seasons(self, synthetic_data):
        f1, l1 = build_tournament_matchups(synthetic_data, [2022])
        f_multi, l_multi = build_tournament_matchups(synthetic_data, [2020, 2021, 2022])
        assert f_multi.shape[0] >= f1.shape[0]

    def test_empty_season_no_crash(self, synthetic_data):
        features, labels = build_tournament_matchups(synthetic_data, [1999])
        assert features.shape[0] == 0

    def test_stats_cache_populated(self, synthetic_data):
        cache = {}
        build_tournament_matchups(synthetic_data, [2022], stats_cache=cache)
        assert 2022 in cache

    def test_lower_id_convention(self, synthetic_data):
        """Labels should follow lower-ID-wins convention."""
        tourney = synthetic_data["MNCAATourneyCompactResults"]
        season_games = tourney[tourney["Season"] == 2022]
        features, labels = build_tournament_matchups(synthetic_data, [2022])

        for i, (_, game) in enumerate(season_games.iterrows()):
            w_id = game["WTeamID"]
            l_id = game["LTeamID"]
            expected_label = 1.0 if min(w_id, l_id) == w_id else 0.0
            assert labels[i] == expected_label
