"""Feature engineering for matchup prediction.

Convention: Team A = lower TeamID, Team B = higher TeamID.
Features are computed as (Team A stat) - (Team B stat) differences.
Label = 1 if Team A wins.
"""

import numpy as np
import pandas as pd


# Ordered list of feature names (must match build_matchup_features output)
FEATURE_NAMES = [
    "SeedDiff",
    "NetEffDiff",
    "MasseyOrdinalDiff",
    "OffEffDiff",
    "DefEffDiff",
    "eFGPctDiff",
    "TOrateDiff",
    "WinPctDiff",
    "ScoringMarginDiff",
    "FTRateDiff",
    "ORPctDiff",
    "RoadWinPctDiff",
    "OppeFGPctDiff",
    "OppTORateDiff",
    "FG3PctDiff",
    "FTPctDiff",
    "AstRateDiff",
    "StlRateDiff",
    "BlkRateDiff",
    "ConsistencyDiff",
    "Last10WinPctDiff",
    "PPGDiff",
    "PAPGDiff",
]

N_FEATURES = len(FEATURE_NAMES)


def _safe_diff(stats: pd.DataFrame, team_a: int, team_b: int, col: str) -> float:
    """Get stat difference (A - B), returning 0.0 if stat missing."""
    if col not in stats.columns:
        return 0.0
    val_a = stats.loc[team_a, col] if team_a in stats.index else np.nan
    val_b = stats.loc[team_b, col] if team_b in stats.index else np.nan
    if pd.isna(val_a) or pd.isna(val_b):
        return 0.0
    return float(val_a - val_b)


def build_matchup_features(team_a: int, team_b: int, stats: pd.DataFrame) -> np.ndarray:
    """Build feature vector for a matchup.

    Args:
        team_a: Lower TeamID (by convention).
        team_b: Higher TeamID.
        stats: Per-team aggregated stats DataFrame (from build_season_team_stats).

    Returns:
        1-D numpy array of shape (N_FEATURES,).
    """
    feats = []

    # Seed difference (lower seed = better, so negate: A seed 1 vs B seed 16 -> -15)
    seed_diff = _safe_diff(stats, team_a, team_b, "Seed")
    feats.append(-seed_diff)  # negate so better seed = positive

    # Massey ordinal (lower rank = better, so negate)
    massey_diff = _safe_diff(stats, team_a, team_b, "MasseyOrdinal")

    # Efficiency metrics (higher = better for off, lower = better for def)
    feats.append(_safe_diff(stats, team_a, team_b, "NetEff"))
    feats.append(-massey_diff)  # negate: lower rank = better
    feats.append(_safe_diff(stats, team_a, team_b, "OffEff"))
    feats.append(-_safe_diff(stats, team_a, team_b, "DefEff"))  # negate: lower def eff = better

    # Shooting and ball control
    feats.append(_safe_diff(stats, team_a, team_b, "eFGPct"))
    feats.append(-_safe_diff(stats, team_a, team_b, "TORate"))  # negate: fewer turnovers = better

    # Overall performance
    feats.append(_safe_diff(stats, team_a, team_b, "WinPct"))
    feats.append(_safe_diff(stats, team_a, team_b, "ScoringMargin"))

    # Secondary stats
    feats.append(_safe_diff(stats, team_a, team_b, "FTRate"))
    feats.append(_safe_diff(stats, team_a, team_b, "ORPct"))
    feats.append(_safe_diff(stats, team_a, team_b, "RoadWinPct"))

    # Opponent stats (lower = better defense, so negate)
    feats.append(-_safe_diff(stats, team_a, team_b, "OppeFGPct"))
    feats.append(_safe_diff(stats, team_a, team_b, "OppTORate"))  # more opp turnovers = better

    # More shooting
    feats.append(_safe_diff(stats, team_a, team_b, "FG3Pct"))
    feats.append(_safe_diff(stats, team_a, team_b, "FTPct"))

    # Other
    feats.append(_safe_diff(stats, team_a, team_b, "AstRate"))
    feats.append(_safe_diff(stats, team_a, team_b, "StlRate"))
    feats.append(_safe_diff(stats, team_a, team_b, "BlkRate"))

    # Consistency & momentum
    feats.append(_safe_diff(stats, team_a, team_b, "Consistency"))
    feats.append(_safe_diff(stats, team_a, team_b, "Last10WinPct"))

    # Scoring
    feats.append(_safe_diff(stats, team_a, team_b, "PPG"))
    feats.append(_safe_diff(stats, team_a, team_b, "PAPG"))

    return np.array(feats, dtype=np.float32)


def build_tournament_matchups(
    data: dict[str, pd.DataFrame],
    seasons: list[int],
    stats_cache: dict[int, pd.DataFrame] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix and labels from historical tournament games.

    Args:
        data: Dict of DataFrames from load_all.
        seasons: List of seasons to include.
        stats_cache: Optional pre-computed {season: stats_df} cache.

    Returns:
        (features, labels) arrays. features shape: (N, N_FEATURES), labels shape: (N,).
    """
    from .data_loader import build_season_team_stats

    tourney = data.get("MNCAATourneyCompactResults")
    if tourney is None:
        tourney = data.get("MNCAATourneyDetailedResults")
    if tourney is None:
        raise KeyError("No tournament results found in data")

    if stats_cache is None:
        stats_cache = {}

    all_features = []
    all_labels = []

    for season in seasons:
        season_games = tourney[tourney["Season"] == season]
        if len(season_games) == 0:
            continue

        if season not in stats_cache:
            stats_cache[season] = build_season_team_stats(data, season)
        stats = stats_cache[season]

        for _, game in season_games.iterrows():
            w_id = int(game["WTeamID"])
            l_id = int(game["LTeamID"])

            # Convention: lower ID = team A
            team_a = min(w_id, l_id)
            team_b = max(w_id, l_id)
            label = 1.0 if w_id == team_a else 0.0

            feats = build_matchup_features(team_a, team_b, stats)
            all_features.append(feats)
            all_labels.append(label)

    if not all_features:
        return np.empty((0, N_FEATURES), dtype=np.float32), np.empty(0, dtype=np.float32)

    features = np.stack(all_features)
    labels = np.array(all_labels, dtype=np.float32)

    # Replace NaN with 0
    features = np.nan_to_num(features, nan=0.0)

    return features, labels
