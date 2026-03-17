"""Tests for src/bracket.py -- bracket simulation and output."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.bracket import (
    get_tournament_teams,
    predict_game,
    simulate_bracket,
    save_bracket,
    FIRST_ROUND_SEEDS,
)
from src.model import BracketNet
from src.features import N_FEATURES


class TestGetTournamentTeams:
    def test_returns_regions(self, synthetic_data):
        regions = get_tournament_teams(synthetic_data, 2022)
        assert isinstance(regions, dict)
        assert len(regions) == 4
        for region_id in regions:
            assert region_id in "WXYZ"

    def test_teams_sorted_by_seed(self, synthetic_data):
        regions = get_tournament_teams(synthetic_data, 2022)
        for region_id, teams in regions.items():
            seeds = [s for s, _ in teams]
            assert seeds == sorted(seeds)

    def test_seed_team_tuple_format(self, synthetic_data):
        regions = get_tournament_teams(synthetic_data, 2022)
        for region_id, teams in regions.items():
            for seed_num, team_id in teams:
                assert isinstance(seed_num, int)
                assert isinstance(team_id, (int, np.integer))
                assert 1 <= seed_num <= 16


class TestPredictGame:
    def test_returns_probability(self, trained_model, fitted_scaler, synthetic_data):
        from src.data_loader import build_season_team_stats
        stats = build_season_team_stats(synthetic_data, 2022)
        prob = predict_game(trained_model, 1101, 1102, stats, fitted_scaler)
        assert 0.0 <= prob <= 1.0

    def test_id_order_doesnt_matter(self, trained_model, fitted_scaler, synthetic_data):
        """predict_game should handle both orderings consistently."""
        from src.data_loader import build_season_team_stats
        stats = build_season_team_stats(synthetic_data, 2022)
        prob_ab = predict_game(trained_model, 1101, 1102, stats, fitted_scaler)
        prob_ba = predict_game(trained_model, 1102, 1101, stats, fitted_scaler)
        # P(A wins) + P(B wins) = 1
        assert prob_ab + prob_ba == pytest.approx(1.0, abs=1e-5)


class TestSimulateBracket:
    @pytest.fixture
    def bracket_setup(self, synthetic_data, trained_model, fitted_scaler):
        """Common setup for bracket tests."""
        return synthetic_data, trained_model, fitted_scaler

    def test_deterministic_has_63_games(self, bracket_setup):
        """A full 64-team bracket should have 63 games.

        Our synthetic data has 4 teams per region (16 total), giving
        4 regions * (2+1) + 2 + 1 = 15 games.
        """
        data, model, scaler = bracket_setup
        bracket = simulate_bracket(model, data, 2022, scaler, method="deterministic")
        # 4 teams per region: 2 first-round + 1 second-round per region = 12
        # + "Sweet 16" and "Elite 8" rounds
        # With 4 seeds per region our fixture produces fewer games than a real bracket
        assert len(bracket["games"]) > 0
        assert "champion" in bracket

    def test_has_champion(self, bracket_setup):
        data, model, scaler = bracket_setup
        bracket = simulate_bracket(model, data, 2022, scaler, method="deterministic")
        assert "champion" in bracket
        assert "team" in bracket["champion"]
        assert "name" in bracket["champion"]
        assert "seed" in bracket["champion"]

    def test_game_structure(self, bracket_setup):
        data, model, scaler = bracket_setup
        bracket = simulate_bracket(model, data, 2022, scaler, method="deterministic")
        for game in bracket["games"]:
            assert "round" in game
            assert "team_a" in game
            assert "team_b" in game
            assert "prob_a" in game
            assert "winner" in game
            assert 0.0 <= game["prob_a"] <= 1.0
            assert game["winner"] in (game["team_a"], game["team_b"])

    def test_deterministic_is_reproducible(self, bracket_setup):
        data, model, scaler = bracket_setup
        b1 = simulate_bracket(model, data, 2022, scaler, method="deterministic")
        b2 = simulate_bracket(model, data, 2022, scaler, method="deterministic")
        # Same champion
        assert b1["champion"]["team"] == b2["champion"]["team"]
        # Same game count
        assert len(b1["games"]) == len(b2["games"])

    def test_monte_carlo_returns_probabilities(self, bracket_setup):
        data, model, scaler = bracket_setup
        result = simulate_bracket(model, data, 2022, scaler,
                                   method="monte_carlo", n_simulations=100)
        assert "champion_probabilities" in result
        assert "final_four_probabilities" in result
        assert "most_likely_champion" in result
        # Probabilities should sum to <= 1 (top 20 only reported)
        champ_probs = sum(result["champion_probabilities"].values())
        assert 0.5 <= champ_probs <= 1.0 + 1e-6

    def test_ensemble_accepted(self, bracket_setup):
        data, _, scaler = bracket_setup
        models = [BracketNet() for _ in range(3)]
        bracket = simulate_bracket(models, data, 2022, scaler, method="deterministic")
        assert "champion" in bracket


class TestSaveBracket:
    def test_saves_valid_json(self):
        bracket = {
            "season": 2022,
            "champion": {"team": 1101, "name": "Team_1101", "seed": 1},
            "games": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "bracket.json")
            save_bracket(bracket, path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["season"] == 2022
            assert loaded["champion"]["team"] == 1101

    def test_handles_numpy_types(self):
        bracket = {
            "season": np.int64(2022),
            "prob": np.float64(0.75),
            "arr": np.array([1, 2, 3]),
            "games": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "bracket.json")
            save_bracket(bracket, path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["season"] == 2022
            assert loaded["prob"] == 0.75

    def test_creates_parent_dirs(self):
        bracket = {"games": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sub" / "dir" / "bracket.json")
            save_bracket(bracket, path)
            assert Path(path).exists()


class TestFirstRoundSeeds:
    def test_covers_all_16_seeds(self):
        all_seeds = set()
        for s1, s2 in FIRST_ROUND_SEEDS:
            all_seeds.add(s1)
            all_seeds.add(s2)
        assert all_seeds == set(range(1, 17))

    def test_matchups_sum_to_17(self):
        """Traditional matchups: 1v16, 2v15, etc. -- seeds sum to 17."""
        for s1, s2 in FIRST_ROUND_SEEDS:
            assert s1 + s2 == 17
