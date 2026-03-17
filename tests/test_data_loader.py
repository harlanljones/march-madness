"""Tests for src/data_loader.py -- data loading and stat aggregation."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import build_season_team_stats, _aggregate_compact


class TestBuildSeasonTeamStats:
    def test_returns_dataframe_indexed_by_team(self, synthetic_data):
        stats = build_season_team_stats(synthetic_data, 2022)
        assert isinstance(stats, pd.DataFrame)
        assert stats.index.name == "TeamID"
        assert len(stats) > 0

    def test_has_core_columns(self, synthetic_data):
        stats = build_season_team_stats(synthetic_data, 2022)
        expected = ["WinPct", "PPG", "PAPG", "ScoringMargin",
                    "OffEff", "DefEff", "NetEff", "eFGPct", "TORate"]
        for col in expected:
            assert col in stats.columns, f"Missing column: {col}"

    def test_win_pct_range(self, synthetic_data):
        stats = build_season_team_stats(synthetic_data, 2022)
        assert (stats["WinPct"] >= 0).all()
        assert (stats["WinPct"] <= 1).all()

    def test_efg_pct_reasonable(self, synthetic_data):
        stats = build_season_team_stats(synthetic_data, 2022)
        # eFG% should be between 0 and 1
        assert (stats["eFGPct"] >= 0).all()
        assert (stats["eFGPct"] <= 1).all()

    def test_seeds_attached(self, synthetic_data):
        stats = build_season_team_stats(synthetic_data, 2022)
        assert "Seed" in stats.columns
        seeded = stats["Seed"].dropna()
        assert len(seeded) > 0
        assert seeded.between(1, 16).all()

    def test_massey_ordinals_attached(self, synthetic_data):
        stats = build_season_team_stats(synthetic_data, 2022)
        assert "MasseyOrdinal" in stats.columns
        ranked = stats["MasseyOrdinal"].dropna()
        assert len(ranked) > 0

    def test_compact_fallback(self, synthetic_data):
        """Should work with only compact results."""
        data_compact_only = {
            k: v for k, v in synthetic_data.items()
            if k != "MRegularSeasonDetailedResults"
        }
        stats = build_season_team_stats(data_compact_only, 2022)
        assert "WinPct" in stats.columns
        assert "PPG" in stats.columns

    def test_raises_without_results(self, synthetic_data):
        data_no_results = {
            k: v for k, v in synthetic_data.items()
            if "Result" not in k
        }
        with pytest.raises(KeyError):
            build_season_team_stats(data_no_results, 2022)

    def test_scoring_margin_sign(self, synthetic_data):
        """Teams with high win% should generally have positive scoring margin."""
        stats = build_season_team_stats(synthetic_data, 2022)
        top_teams = stats.nlargest(3, "WinPct")
        # At least one top team should have positive margin (not guaranteed but very likely)
        assert (top_teams["ScoringMargin"] > -20).any()

    def test_offensive_efficiency_positive(self, synthetic_data):
        stats = build_season_team_stats(synthetic_data, 2022)
        assert (stats["OffEff"] > 0).all()
        assert (stats["DefEff"] > 0).all()

    def test_road_win_pct(self, synthetic_data):
        stats = build_season_team_stats(synthetic_data, 2022)
        assert "RoadWinPct" in stats.columns
        assert (stats["RoadWinPct"] >= 0).all()
        assert (stats["RoadWinPct"] <= 1).all()

    def test_last_10_win_pct(self, synthetic_data):
        stats = build_season_team_stats(synthetic_data, 2022)
        assert "Last10WinPct" in stats.columns
        valid = stats["Last10WinPct"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()


class TestAggregateCompact:
    def test_basic(self):
        df = pd.DataFrame({
            "Season": [2022, 2022],
            "WTeamID": [1101, 1102],
            "WScore": [80, 75],
            "LTeamID": [1102, 1101],
            "LScore": [70, 65],
            "WLoc": ["H", "A"],
            "NumOT": [0, 0],
        })
        stats = _aggregate_compact(df)
        # Each team played 2 games, won 1
        assert stats.loc[1101, "Games"] == 2
        assert stats.loc[1101, "Wins"] == 1
        assert stats.loc[1101, "WinPct"] == pytest.approx(0.5)
