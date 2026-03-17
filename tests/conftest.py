"""Shared fixtures for March Madness tests.

Builds synthetic DataFrames that mirror Kaggle's schema so tests run
without real data on disk.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.model import BracketNet
from src.features import N_FEATURES


# ---------------------------------------------------------------------------
# Minimal synthetic Kaggle-like data
# ---------------------------------------------------------------------------

TEAM_IDS = [
    # 64 teams total: 16 per region
    1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108,
    1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116,
    1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208,
    1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216,
    1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308,
    1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316,
    1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408,
    1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416,
]

TEAM_NAMES = {tid: f"Team_{tid}" for tid in TEAM_IDS}


def _make_teams_df() -> pd.DataFrame:
    return pd.DataFrame({
        "TeamID": TEAM_IDS,
        "TeamName": [TEAM_NAMES[t] for t in TEAM_IDS],
    })


def _make_detailed_results(season: int, n_games: int = 300) -> pd.DataFrame:
    """Generate synthetic regular-season detailed results."""
    rng = np.random.RandomState(season)
    rows = []
    for _ in range(n_games):
        t1, t2 = rng.choice(TEAM_IDS, size=2, replace=False)
        w_score = rng.randint(60, 100)
        l_score = rng.randint(50, w_score)
        loc = rng.choice(["H", "A", "N"])

        row = {
            "Season": season, "DayNum": rng.randint(1, 133),
            "WTeamID": t1, "WScore": w_score,
            "LTeamID": t2, "LScore": l_score,
            "WLoc": loc, "NumOT": 0,
        }
        # Detailed box-score columns for winner
        for prefix, base_fg in [("W", rng.randint(20, 35)), ("L", rng.randint(18, 30))]:
            row[f"{prefix}FGM"] = base_fg
            row[f"{prefix}FGA"] = base_fg + rng.randint(15, 30)
            row[f"{prefix}FGM3"] = rng.randint(3, 12)
            row[f"{prefix}FGA3"] = row[f"{prefix}FGM3"] + rng.randint(5, 15)
            row[f"{prefix}FTM"] = rng.randint(8, 20)
            row[f"{prefix}FTA"] = row[f"{prefix}FTM"] + rng.randint(0, 8)
            row[f"{prefix}OR"] = rng.randint(5, 15)
            row[f"{prefix}DR"] = rng.randint(15, 30)
            row[f"{prefix}Ast"] = rng.randint(8, 20)
            row[f"{prefix}TO"] = rng.randint(8, 18)
            row[f"{prefix}Stl"] = rng.randint(3, 10)
            row[f"{prefix}Blk"] = rng.randint(1, 7)
            row[f"{prefix}PF"] = rng.randint(10, 25)
        rows.append(row)
    return pd.DataFrame(rows)


def _make_compact_results(season: int, n_games: int = 300) -> pd.DataFrame:
    """Generate synthetic regular-season compact results."""
    rng = np.random.RandomState(season)
    rows = []
    for _ in range(n_games):
        t1, t2 = rng.choice(TEAM_IDS, size=2, replace=False)
        w_score = rng.randint(60, 100)
        l_score = rng.randint(50, w_score)
        rows.append({
            "Season": season, "DayNum": rng.randint(1, 133),
            "WTeamID": t1, "WScore": w_score,
            "LTeamID": t2, "LScore": l_score,
            "WLoc": rng.choice(["H", "A", "N"]), "NumOT": 0,
        })
    return pd.DataFrame(rows)


def _make_tourney_seeds(season: int) -> pd.DataFrame:
    """Assign 64 teams as seeds across 4 regions (16 teams each)."""
    rows = []
    for i, (region, ids) in enumerate(zip(
        ["W", "X", "Y", "Z"],
        [TEAM_IDS[0:16], TEAM_IDS[16:32], TEAM_IDS[32:48], TEAM_IDS[48:64]],
    )):
        for seed_num, tid in zip(range(1, 17), ids):
            rows.append({"Season": season, "Seed": f"{region}{seed_num:02d}", "TeamID": tid})
    return pd.DataFrame(rows)


def _make_tourney_results(season: int) -> pd.DataFrame:
    """Generate synthetic tournament results across all regions."""
    rng = np.random.RandomState(season + 999)
    rows = []

    # Simulate some tournament games from each region (higher seed wins)
    for region_start in range(0, 64, 16):
        region_teams = TEAM_IDS[region_start:region_start + 16]
        # First round: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
        matchups = [(0, 15), (7, 8), (4, 11), (3, 12), (5, 10), (2, 13), (6, 9), (1, 14)]
        winners = []
        for s1_idx, s2_idx in matchups:
            w = region_teams[s1_idx]
            l = region_teams[s2_idx]
            w_score = rng.randint(65, 95)
            l_score = rng.randint(50, w_score)
            rows.append({"Season": season, "DayNum": 136, "WTeamID": w,
                         "WScore": w_score, "LTeamID": l, "LScore": l_score,
                         "WLoc": "N", "NumOT": 0})
            winners.append(w)
        # Second round
        r2_winners = []
        for i in range(0, 8, 2):
            w, l = winners[i], winners[i + 1]
            w_score = rng.randint(65, 95)
            l_score = rng.randint(50, w_score)
            rows.append({"Season": season, "DayNum": 138, "WTeamID": w,
                         "WScore": w_score, "LTeamID": l, "LScore": l_score,
                         "WLoc": "N", "NumOT": 0})
            r2_winners.append(w)
        # Sweet 16
        s16_winners = []
        for i in range(0, 4, 2):
            w, l = r2_winners[i], r2_winners[i + 1]
            w_score = rng.randint(65, 95)
            l_score = rng.randint(50, w_score)
            rows.append({"Season": season, "DayNum": 143, "WTeamID": w,
                         "WScore": w_score, "LTeamID": l, "LScore": l_score,
                         "WLoc": "N", "NumOT": 0})
            s16_winners.append(w)
        # Elite 8
        w, l = s16_winners[0], s16_winners[1]
        w_score = rng.randint(65, 95)
        l_score = rng.randint(50, w_score)
        rows.append({"Season": season, "DayNum": 145, "WTeamID": w,
                     "WScore": w_score, "LTeamID": l, "LScore": l_score,
                     "WLoc": "N", "NumOT": 0})

    return pd.DataFrame(rows)


def _make_massey_ordinals(season: int) -> pd.DataFrame:
    """Generate synthetic Massey ordinals (POM system)."""
    rows = []
    for tid in TEAM_IDS:
        rows.append({
            "Season": season,
            "RankingDayNum": 128,
            "SystemName": "POM",
            "TeamID": tid,
            "OrdinalRank": TEAM_IDS.index(tid) + 1,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_data() -> dict[str, pd.DataFrame]:
    """Full synthetic dataset dict mimicking load_all() output.

    Contains seasons 2020, 2021, 2022 for training and 2023 for validation.
    """
    seasons = [2020, 2021, 2022, 2023]

    detailed_frames = [_make_detailed_results(s) for s in seasons]
    compact_frames = [_make_compact_results(s) for s in seasons]
    seed_frames = [_make_tourney_seeds(s) for s in seasons]
    tourney_frames = [_make_tourney_results(s) for s in seasons]
    massey_frames = [_make_massey_ordinals(s) for s in seasons]

    return {
        "MTeams": _make_teams_df(),
        "MRegularSeasonDetailedResults": pd.concat(detailed_frames, ignore_index=True),
        "MRegularSeasonCompactResults": pd.concat(compact_frames, ignore_index=True),
        "MNCAATourneySeeds": pd.concat(seed_frames, ignore_index=True),
        "MNCAATourneyCompactResults": pd.concat(tourney_frames, ignore_index=True),
        "MMasseyOrdinals": pd.concat(massey_frames, ignore_index=True),
    }


@pytest.fixture
def trained_model() -> BracketNet:
    """Return an untrained BracketNet (random weights) for structural tests."""
    torch.manual_seed(42)
    return BracketNet(input_dim=N_FEATURES)


@pytest.fixture
def fitted_scaler(synthetic_data):
    """Return a fitted StandardScaler from synthetic training data."""
    from sklearn.preprocessing import StandardScaler
    from src.features import build_tournament_matchups

    feats, _ = build_tournament_matchups(synthetic_data, [2020, 2021, 2022])
    scaler = StandardScaler()
    scaler.fit(feats)
    return scaler
