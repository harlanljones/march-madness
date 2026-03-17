"""Bracket simulation and output generation."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .utils import parse_seed, build_team_lookup
from .features import build_matchup_features
from .model import BracketNet
from .data_loader import build_season_team_stats


# Standard first-round matchups by seed
FIRST_ROUND_SEEDS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
REGIONS = ["W", "X", "Y", "Z"]
ROUND_NAMES = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]


def get_tournament_teams(data: dict[str, pd.DataFrame], season: int) -> dict[str, list[tuple[int, int]]]:
    """Get tournament teams organized by region.

    Returns:
        Dict mapping region -> list of (seed_num, team_id) sorted by seed.
    """
    seeds_df = data["MNCAATourneySeeds"]
    season_seeds = seeds_df[seeds_df["Season"] == season]

    regions = {}
    for _, row in season_seeds.iterrows():
        region, seed_num = parse_seed(row["Seed"])
        team_id = row["TeamID"]
        if region not in regions:
            regions[region] = []
        regions[region].append((seed_num, team_id))

    for region in regions:
        regions[region].sort(key=lambda x: x[0])

    return regions


def predict_game(
    model: BracketNet,
    team_a: int,
    team_b: int,
    stats: pd.DataFrame,
    scaler: StandardScaler,
    device: str = "cpu",
) -> float:
    """Predict probability that lower-ID team wins."""
    t_a = min(team_a, team_b)
    t_b = max(team_a, team_b)

    feats = build_matchup_features(t_a, t_b, stats)
    feats_scaled = scaler.transform(feats.reshape(1, -1))

    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(feats_scaled, dtype=torch.float32).to(device)
        prob = model(tensor).item()

    # prob is P(lower_id wins). If team_a != lower_id, flip.
    if team_a > team_b:
        prob = 1.0 - prob

    return prob  # P(team_a wins)


def predict_game_ensemble(
    models: list[BracketNet],
    team_a: int,
    team_b: int,
    stats: pd.DataFrame,
    scaler: StandardScaler,
    device: str = "cpu",
) -> float:
    """Average prediction across ensemble."""
    probs = [predict_game(m, team_a, team_b, stats, scaler, device) for m in models]
    return np.mean(probs)


def simulate_bracket(
    model: BracketNet | list[BracketNet],
    data: dict[str, pd.DataFrame],
    season: int,
    scaler: StandardScaler,
    method: str = "deterministic",
    n_simulations: int = 10000,
    device: str = "cpu",
) -> dict:
    """Simulate the full tournament bracket.

    Args:
        model: Single model or list of models (ensemble).
        method: 'deterministic', 'probability', or 'monte_carlo'.
        n_simulations: Number of simulations for monte_carlo method.

    Returns:
        Dict with bracket results.
    """
    models = model if isinstance(model, list) else [model]
    regions = get_tournament_teams(data, season)
    stats = build_season_team_stats(data, season)
    team_names = build_team_lookup(data["MTeams"])

    def get_prob(team_a, team_b):
        return predict_game_ensemble(models, team_a, team_b, stats, scaler, device)

    def pick_winner(team_a, team_b, prob_a):
        if method == "deterministic":
            return team_a if prob_a >= 0.5 else team_b
        elif method == "monte_carlo":
            return team_a if np.random.random() < prob_a else team_b
        return team_a  # shouldn't reach here

    if method == "monte_carlo":
        return _monte_carlo_bracket(models, regions, stats, scaler, team_names,
                                     n_simulations, device)

    # Deterministic or probability mode
    bracket = {"method": method, "season": season, "regions": {}, "final_four": [], "championship": {}, "games": []}

    final_four_teams = []

    for region_id, teams in regions.items():
        seed_to_team = {seed: tid for seed, tid in teams}
        round_teams = []

        # Build first round from seed matchups
        for s1, s2 in FIRST_ROUND_SEEDS:
            if s1 in seed_to_team and s2 in seed_to_team:
                round_teams.append((seed_to_team[s1], s1, seed_to_team[s2], s2))

        region_results = {"region": region_id, "rounds": []}
        current_winners = []

        # First round
        for team_a, seed_a, team_b, seed_b in round_teams:
            prob = get_prob(team_a, team_b)
            winner = pick_winner(team_a, team_b, prob)
            winner_seed = seed_a if winner == team_a else seed_b
            loser = team_b if winner == team_a else team_a

            game = {
                "round": "Round of 64",
                "region": region_id,
                "team_a": team_a, "team_a_name": team_names.get(team_a, str(team_a)), "seed_a": seed_a,
                "team_b": team_b, "team_b_name": team_names.get(team_b, str(team_b)), "seed_b": seed_b,
                "prob_a": round(prob, 4),
                "winner": winner, "winner_name": team_names.get(winner, str(winner)), "winner_seed": winner_seed,
            }
            bracket["games"].append(game)
            current_winners.append((winner, winner_seed))

        region_results["rounds"].append(current_winners[:])

        # Subsequent rounds within region (R32, S16, E8)
        for round_name in ["Round of 32", "Sweet 16", "Elite 8"]:
            next_winners = []
            for i in range(0, len(current_winners), 2):
                team_a, seed_a = current_winners[i]
                team_b, seed_b = current_winners[i + 1]
                prob = get_prob(team_a, team_b)
                winner = pick_winner(team_a, team_b, prob)
                winner_seed = seed_a if winner == team_a else seed_b

                game = {
                    "round": round_name,
                    "region": region_id,
                    "team_a": team_a, "team_a_name": team_names.get(team_a, str(team_a)), "seed_a": seed_a,
                    "team_b": team_b, "team_b_name": team_names.get(team_b, str(team_b)), "seed_b": seed_b,
                    "prob_a": round(prob, 4),
                    "winner": winner, "winner_name": team_names.get(winner, str(winner)), "winner_seed": winner_seed,
                }
                bracket["games"].append(game)
                next_winners.append((winner, winner_seed))

            current_winners = next_winners
            region_results["rounds"].append(current_winners[:])

        bracket["regions"][region_id] = region_results
        final_four_teams.append(current_winners[0])

    # Final Four
    region_keys = sorted(bracket["regions"].keys())
    # Semis: region pairs (0 vs 1, 2 vs 3 by convention)
    semis = [(0, 1), (2, 3)]
    championship_teams = []

    for i, j in semis:
        team_a, seed_a = final_four_teams[i]
        team_b, seed_b = final_four_teams[j]
        prob = get_prob(team_a, team_b)
        winner = pick_winner(team_a, team_b, prob)
        winner_seed = seed_a if winner == team_a else seed_b

        game = {
            "round": "Final Four",
            "team_a": team_a, "team_a_name": team_names.get(team_a, str(team_a)), "seed_a": seed_a,
            "team_b": team_b, "team_b_name": team_names.get(team_b, str(team_b)), "seed_b": seed_b,
            "prob_a": round(prob, 4),
            "winner": winner, "winner_name": team_names.get(winner, str(winner)), "winner_seed": winner_seed,
        }
        bracket["games"].append(game)
        championship_teams.append((winner, winner_seed))

    # Championship
    team_a, seed_a = championship_teams[0]
    team_b, seed_b = championship_teams[1]
    prob = get_prob(team_a, team_b)
    winner = pick_winner(team_a, team_b, prob)
    winner_seed = seed_a if winner == team_a else seed_b

    game = {
        "round": "Championship",
        "team_a": team_a, "team_a_name": team_names.get(team_a, str(team_a)), "seed_a": seed_a,
        "team_b": team_b, "team_b_name": team_names.get(team_b, str(team_b)), "seed_b": seed_b,
        "prob_a": round(prob, 4),
        "winner": winner, "winner_name": team_names.get(winner, str(winner)), "winner_seed": winner_seed,
    }
    bracket["games"].append(game)
    bracket["champion"] = {"team": winner, "name": team_names.get(winner, str(winner)), "seed": winner_seed}

    return bracket


def _precompute_all_probs(models, all_team_ids, stats, scaler, device):
    """Batch-compute win probabilities for every possible matchup.

    Builds one large feature matrix covering all (lower_id, higher_id) pairs
    among tournament teams, runs a single batched forward pass per model,
    and returns a {(t_a, t_b): prob_a_wins} lookup dict.

    This means the GPU is used once rather than once per simulated game,
    making Monte Carlo ~100x faster on GPU.
    """
    ids = sorted(set(all_team_ids))
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i+1:]]

    if not pairs:
        return {}

    # Build feature matrix
    feats = np.stack([build_matchup_features(a, b, stats) for a, b in pairs])
    feats_scaled = scaler.transform(feats)
    tensor = torch.tensor(feats_scaled, dtype=torch.float32).to(device)

    # Average ensemble predictions in one pass each
    with torch.no_grad():
        probs = torch.stack([m(tensor) for m in models]).mean(dim=0).cpu().numpy()

    return {pair: float(p) for pair, p in zip(pairs, probs)}


def _monte_carlo_bracket(models, regions, stats, scaler, team_names, n_sims, device):
    """Run Monte Carlo bracket simulation using precomputed probability lookup."""
    # Collect all tournament team IDs
    all_team_ids = [tid for teams in regions.values() for _, tid in teams]

    # Precompute all pairwise probabilities in one GPU batch
    print(f"  Precomputing {len(all_team_ids)*(len(all_team_ids)-1)//2} matchup probabilities...")
    prob_table = _precompute_all_probs(models, all_team_ids, stats, scaler, device)

    def get_prob(t_a, t_b):
        """P(t_a wins), using the precomputed table."""
        lo, hi = min(t_a, t_b), max(t_a, t_b)
        p = prob_table.get((lo, hi), 0.5)
        return p if t_a == lo else 1.0 - p

    champion_counts = Counter()
    final_four_counts = Counter()

    # Draw all random numbers upfront for speed
    # Max games per sim: 63. We'll draw generously and index in.
    rng_vals = np.random.random((n_sims, 63))

    for sim in range(n_sims):
        rng_idx = 0
        final_four_teams = []

        for region_id, teams in regions.items():
            seed_to_team = {seed: tid for seed, tid in teams}
            current = []
            for s1, s2 in FIRST_ROUND_SEEDS:
                if s1 in seed_to_team and s2 in seed_to_team:
                    current.append(seed_to_team[s1])
                    current.append(seed_to_team[s2])

            while len(current) > 1:
                next_round = []
                for i in range(0, len(current), 2):
                    prob = get_prob(current[i], current[i+1])
                    winner = current[i] if rng_vals[sim, rng_idx] < prob else current[i+1]
                    rng_idx += 1
                    next_round.append(winner)
                current = next_round

            final_four_teams.append(current[0])
            final_four_counts[current[0]] += 1

        # Final Four
        champ_teams = []
        for i, j in [(0, 1), (2, 3)]:
            prob = get_prob(final_four_teams[i], final_four_teams[j])
            winner = final_four_teams[i] if rng_vals[sim, rng_idx] < prob else final_four_teams[j]
            rng_idx += 1
            champ_teams.append(winner)

        prob = get_prob(champ_teams[0], champ_teams[1])
        champion = champ_teams[0] if rng_vals[sim, rng_idx] < prob else champ_teams[1]
        champion_counts[champion] += 1

    results = {
        "method": "monte_carlo",
        "n_simulations": n_sims,
        "champion_probabilities": {
            team_names.get(tid, str(tid)): count / n_sims
            for tid, count in champion_counts.most_common(20)
        },
        "final_four_probabilities": {
            team_names.get(tid, str(tid)): count / n_sims
            for tid, count in final_four_counts.most_common(20)
        },
        "most_likely_champion": team_names.get(
            champion_counts.most_common(1)[0][0],
            str(champion_counts.most_common(1)[0][0])
        ),
    }

    return results


def print_bracket(bracket: dict):
    """Pretty-print bracket to terminal."""
    if bracket.get("method") == "monte_carlo":
        _print_monte_carlo(bracket)
        return

    print("\n" + "=" * 70)
    print(f"  MARCH MADNESS {bracket.get('season', '')} BRACKET")
    print("=" * 70)

    # Group games by round
    rounds = {}
    for game in bracket["games"]:
        rnd = game["round"]
        if rnd not in rounds:
            rounds[rnd] = []
        rounds[rnd].append(game)

    for round_name in ROUND_NAMES:
        if round_name not in rounds:
            continue
        print(f"\n--- {round_name} ---")
        for g in rounds[round_name]:
            region = g.get("region", "")
            prefix = f"[{region}] " if region else ""
            prob_str = f"{g['prob_a']:.1%}" if g["prob_a"] >= 0.5 else f"{1-g['prob_a']:.1%}"
            winner_marker = ">>>" if g["winner"] == g["team_a"] else "   "
            loser_marker = "   " if g["winner"] == g["team_a"] else ">>>"

            print(f"  {prefix}{winner_marker} ({g['seed_a']:2d}) {g['team_a_name']:<25s}")
            print(f"  {prefix}{loser_marker} ({g['seed_b']:2d}) {g['team_b_name']:<25s}  [{prob_str}]")
            print()

    if "champion" in bracket:
        print("=" * 70)
        print(f"  CHAMPION: ({bracket['champion']['seed']}) {bracket['champion']['name']}")
        print("=" * 70)


def _print_monte_carlo(results: dict):
    """Print Monte Carlo simulation results."""
    print("\n" + "=" * 70)
    print(f"  MONTE CARLO SIMULATION ({results['n_simulations']:,} runs)")
    print("=" * 70)

    print("\n  Championship Probabilities:")
    for name, prob in results["champion_probabilities"].items():
        bar = "#" * int(prob * 50)
        print(f"    {name:<25s} {prob:6.1%}  {bar}")

    print(f"\n  Most Likely Champion: {results['most_likely_champion']}")

    print("\n  Final Four Probabilities:")
    for name, prob in list(results["final_four_probabilities"].items())[:10]:
        bar = "#" * int(prob * 50)
        print(f"    {name:<25s} {prob:6.1%}  {bar}")

    print("=" * 70)


def save_bracket(bracket: dict, path: str):
    """Save bracket to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert any numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w") as f:
        json.dump(bracket, f, indent=2, default=convert)
    print(f"Bracket saved to {path}")
