"""Load and merge raw Kaggle CSVs into usable DataFrames."""

import os
from pathlib import Path

import pandas as pd


# Files we expect in the Kaggle dataset (M = Men's tournament)
EXPECTED_FILES = [
    "MTeams",
    "MSeasons",
    "MRegularSeasonCompactResults",
    "MRegularSeasonDetailedResults",
    "MNCAATourneyCompactResults",
    "MNCAATourneyDetailedResults",
    "MNCAATourneySeeds",
    "MMasseyOrdinals",
    "MNCAATourneySlots",
    "MNCAATourneySeedRoundSlots",
    "MConferenceTourneyGames",
    "MGameCities",
    "MSecondaryTourneyCompactResults",
    "MSecondaryTourneyTeams",
    "MTeamCoaches",
    "MTeamConferences",
]


def load_all(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load all available Kaggle CSVs from data_dir.

    Returns:
        Dict mapping filename stem (e.g. 'MTeams') to DataFrame.
    """
    data_dir = Path(data_dir)
    data = {}
    for f in sorted(data_dir.glob("*.csv")):
        key = f.stem
        data[key] = pd.read_csv(f)
    if not data:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    missing = [f for f in EXPECTED_FILES if f not in data]
    if missing:
        print(f"Warning: missing expected files: {missing}")

    return data


def build_season_team_stats(data: dict[str, pd.DataFrame], season: int) -> pd.DataFrame:
    """Build per-team aggregated stats for a given season.

    Uses regular season detailed results to compute offensive and
    defensive efficiency metrics per team.

    Returns:
        DataFrame indexed by TeamID with aggregated stats.
    """
    detailed = data.get("MRegularSeasonDetailedResults")
    compact = data.get("MRegularSeasonCompactResults")

    if detailed is None and compact is None:
        raise KeyError("Need MRegularSeasonDetailedResults or MRegularSeasonCompactResults")

    # Use detailed results if available
    if detailed is not None:
        df = detailed[detailed["Season"] == season].copy()
        stats = _aggregate_detailed(df)
    else:
        df = compact[compact["Season"] == season].copy()
        stats = _aggregate_compact(df)

    # Add seeds if available
    seeds = data.get("MNCAATourneySeeds")
    if seeds is not None:
        season_seeds = seeds[seeds["Season"] == season].copy()
        from .utils import seed_int
        season_seeds["SeedNum"] = season_seeds["Seed"].apply(seed_int)
        seed_map = dict(zip(season_seeds["TeamID"], season_seeds["SeedNum"]))
        stats["Seed"] = stats.index.map(seed_map)

    # Add Massey ordinals (use last available day, POM ranking)
    ordinals = data.get("MMasseyOrdinals")
    if ordinals is not None:
        szn_ord = ordinals[(ordinals["Season"] == season) & (ordinals["SystemName"] == "POM")]
        if len(szn_ord) > 0:
            last_day = szn_ord["RankingDayNum"].max()
            final_ranks = szn_ord[szn_ord["RankingDayNum"] == last_day]
            rank_map = dict(zip(final_ranks["TeamID"], final_ranks["OrdinalRank"]))
            stats["MasseyOrdinal"] = stats.index.map(rank_map)

    return stats


def _aggregate_detailed(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detailed results into per-team stats."""
    # Winning team stats
    w_stats = df.rename(columns=lambda c: c[1:] if c.startswith("W") and c != "WLoc" else c)
    w_stats["Won"] = 1
    w_stats["Poss"] = w_stats["FGA"] - w_stats["OR"] + w_stats["TO"] + 0.475 * w_stats["FTA"]
    w_stats["OppPoss"] = w_stats.eval("LFGA - LOR + LTO + 0.475 * LFTA")
    w_stats["PtsAllowed"] = w_stats["LScore"]
    w_stats["OppOR"] = w_stats["LOR"]
    w_stats["OppDR"] = w_stats["LDR"]
    w_stats["OppFGA"] = w_stats["LFGA"]
    w_stats["OppFGM"] = w_stats["LFGM"]
    w_stats["OppFGA3"] = w_stats["LFGA3"]
    w_stats["OppFGM3"] = w_stats["LFGM3"]
    w_stats["OppFTA"] = w_stats["LFTA"]
    w_stats["OppFTM"] = w_stats["LFTM"]
    w_stats["OppTO"] = w_stats["LTO"]
    w_stats["IsHome"] = (df["WLoc"] == "H").astype(int)
    w_stats["IsAway"] = (df["WLoc"] == "A").astype(int)

    # Losing team stats (mirror)
    l_stats = df.copy()
    l_stats["TeamID"] = df["LTeamID"]
    l_stats["Score"] = df["LScore"]
    l_stats["Won"] = 0
    for col in ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]:
        l_stats[col] = df[f"L{col}"]
    l_stats["Poss"] = l_stats["FGA"] - l_stats["OR"] + l_stats["TO"] + 0.475 * l_stats["FTA"]
    l_stats["OppPoss"] = w_stats["Poss"].values
    l_stats["PtsAllowed"] = df["WScore"]
    l_stats["OppOR"] = df["WOR"]
    l_stats["OppDR"] = df["WDR"]
    l_stats["OppFGA"] = df["WFGA"]
    l_stats["OppFGM"] = df["WFGM"]
    l_stats["OppFGA3"] = df["WFGA3"]
    l_stats["OppFGM3"] = df["WFGM3"]
    l_stats["OppFTA"] = df["WFTA"]
    l_stats["OppFTM"] = df["WFTM"]
    l_stats["OppTO"] = df["WTO"]
    l_stats["IsHome"] = (df["WLoc"] == "A").astype(int)
    l_stats["IsAway"] = (df["WLoc"] == "H").astype(int)

    # Combine and aggregate
    keep_cols = ["TeamID", "Score", "Won", "FGM", "FGA", "FGM3", "FGA3",
                 "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF",
                 "Poss", "OppPoss", "PtsAllowed", "OppOR", "OppDR",
                 "OppFGA", "OppFGM", "OppFGA3", "OppFGM3", "OppFTA", "OppFTM", "OppTO",
                 "IsHome", "IsAway"]
    all_games = pd.concat([w_stats[keep_cols], l_stats[keep_cols]], ignore_index=True)

    agg = all_games.groupby("TeamID").agg(
        Games=("Won", "count"),
        Wins=("Won", "sum"),
        TotalPts=("Score", "sum"),
        TotalPtsAllowed=("PtsAllowed", "sum"),
        TotalFGM=("FGM", "sum"),
        TotalFGA=("FGA", "sum"),
        TotalFGM3=("FGM3", "sum"),
        TotalFGA3=("FGA3", "sum"),
        TotalFTM=("FTM", "sum"),
        TotalFTA=("FTA", "sum"),
        TotalOR=("OR", "sum"),
        TotalDR=("DR", "sum"),
        TotalAst=("Ast", "sum"),
        TotalTO=("TO", "sum"),
        TotalStl=("Stl", "sum"),
        TotalBlk=("Blk", "sum"),
        TotalPF=("PF", "sum"),
        TotalPoss=("Poss", "sum"),
        TotalOppPoss=("OppPoss", "sum"),
        TotalOppOR=("OppOR", "sum"),
        TotalOppDR=("OppDR", "sum"),
        TotalOppFGA=("OppFGA", "sum"),
        TotalOppFGM=("OppFGM", "sum"),
        TotalOppFGA3=("OppFGA3", "sum"),
        TotalOppFGM3=("OppFGM3", "sum"),
        TotalOppFTA=("OppFTA", "sum"),
        TotalOppFTM=("OppFTM", "sum"),
        TotalOppTO=("OppTO", "sum"),
        HomeGames=("IsHome", "sum"),
        AwayGames=("IsAway", "sum"),
    )

    # Derived per-game / rate stats
    g = agg["Games"]
    agg["WinPct"] = agg["Wins"] / g
    agg["PPG"] = agg["TotalPts"] / g
    agg["PAPG"] = agg["TotalPtsAllowed"] / g
    agg["ScoringMargin"] = agg["PPG"] - agg["PAPG"]
    agg["OffEff"] = agg["TotalPts"] / agg["TotalPoss"] * 100  # pts per 100 poss
    agg["DefEff"] = agg["TotalPtsAllowed"] / agg["TotalOppPoss"] * 100
    agg["NetEff"] = agg["OffEff"] - agg["DefEff"]
    agg["eFGPct"] = (agg["TotalFGM"] + 0.5 * agg["TotalFGM3"]) / agg["TotalFGA"]
    agg["TORate"] = agg["TotalTO"] / agg["TotalPoss"]
    agg["FTRate"] = agg["TotalFTM"] / agg["TotalFGA"]
    agg["ORPct"] = agg["TotalOR"] / (agg["TotalOR"] + agg["TotalOppDR"])
    agg["OppeFGPct"] = (agg["TotalOppFGM"] + 0.5 * agg["TotalOppFGM3"]) / agg["TotalOppFGA"]
    agg["OppTORate"] = agg["TotalOppTO"] / agg["TotalOppPoss"]
    agg["FG3Pct"] = agg["TotalFGM3"] / agg["TotalFGA3"].replace(0, 1)
    agg["FTPct"] = agg["TotalFTM"] / agg["TotalFTA"].replace(0, 1)
    agg["AstRate"] = agg["TotalAst"] / agg["TotalFGM"].replace(0, 1)
    agg["StlRate"] = agg["TotalStl"] / agg["TotalOppPoss"] * 100
    agg["BlkRate"] = agg["TotalBlk"] / agg["TotalOppFGA"].replace(0, 1) * 100

    # Road win percentage
    away_games = all_games[all_games["IsAway"] == 1]
    if len(away_games) > 0:
        road_wins = away_games.groupby("TeamID")["Won"].agg(["sum", "count"])
        road_wins["RoadWinPct"] = road_wins["sum"] / road_wins["count"]
        agg["RoadWinPct"] = road_wins["RoadWinPct"]
        agg["RoadWinPct"] = agg["RoadWinPct"].fillna(agg["WinPct"])

    # Scoring consistency (std dev of scoring margin)
    margins = all_games.copy()
    margins["Margin"] = margins["Score"] - margins["PtsAllowed"]
    margin_std = margins.groupby("TeamID")["Margin"].std()
    agg["Consistency"] = -margin_std  # negative so higher = more consistent

    # Last 10 games momentum
    all_games_sorted = all_games.sort_values("TeamID")
    last10 = all_games.groupby("TeamID").tail(10)
    last10_winpct = last10.groupby("TeamID")["Won"].mean()
    agg["Last10WinPct"] = last10_winpct

    return agg


def _aggregate_compact(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate compact results (fallback when detailed not available)."""
    w = df[["Season", "WTeamID", "WScore", "LScore"]].copy()
    w.columns = ["Season", "TeamID", "Score", "OppScore"]
    w["Won"] = 1

    l = df[["Season", "LTeamID", "LScore", "WScore"]].copy()
    l.columns = ["Season", "TeamID", "Score", "OppScore"]
    l["Won"] = 0

    all_games = pd.concat([w, l], ignore_index=True)

    agg = all_games.groupby("TeamID").agg(
        Games=("Won", "count"),
        Wins=("Won", "sum"),
        TotalPts=("Score", "sum"),
        TotalPtsAllowed=("OppScore", "sum"),
    )
    g = agg["Games"]
    agg["WinPct"] = agg["Wins"] / g
    agg["PPG"] = agg["TotalPts"] / g
    agg["PAPG"] = agg["TotalPtsAllowed"] / g
    agg["ScoringMargin"] = agg["PPG"] - agg["PAPG"]

    return agg
