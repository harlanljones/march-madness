"""Tests for src/utils.py -- seed parsing and team lookups."""

import pandas as pd
import pytest

from src.utils import parse_seed, seed_int, build_team_lookup


class TestParseSeed:
    def test_standard_seeds(self):
        assert parse_seed("W01") == ("W", 1)
        assert parse_seed("X16") == ("X", 16)
        assert parse_seed("Y08") == ("Y", 8)
        assert parse_seed("Z15") == ("Z", 15)

    def test_play_in_suffix(self):
        """Seeds like W16a / W16b have a play-in suffix."""
        assert parse_seed("W16a") == ("W", 16)
        assert parse_seed("X11b") == ("X", 11)

    def test_whitespace_stripped(self):
        assert parse_seed("  W01  ") == ("W", 1)

    def test_invalid_region_raises(self):
        with pytest.raises(ValueError, match="Cannot parse seed"):
            parse_seed("A01")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_seed("")

    def test_numeric_only_raises(self):
        with pytest.raises(ValueError):
            parse_seed("01")

    def test_all_regions(self):
        for region in "WXYZ":
            r, s = parse_seed(f"{region}08")
            assert r == region
            assert s == 8

    def test_all_seed_numbers(self):
        for n in range(1, 17):
            _, s = parse_seed(f"W{n:02d}")
            assert s == n


class TestSeedInt:
    def test_returns_int(self):
        assert seed_int("W01") == 1
        assert seed_int("Z16a") == 16

    def test_type(self):
        assert isinstance(seed_int("W01"), int)


class TestBuildTeamLookup:
    def test_basic(self):
        df = pd.DataFrame({"TeamID": [1101, 1102], "TeamName": ["Duke", "UNC"]})
        lookup = build_team_lookup(df)
        assert lookup == {1101: "Duke", 1102: "UNC"}

    def test_empty(self):
        df = pd.DataFrame({"TeamID": [], "TeamName": []})
        assert build_team_lookup(df) == {}
