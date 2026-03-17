"""Helpers for seed parsing and team lookups."""

import re


def parse_seed(seed_str: str) -> tuple[str, int]:
    """Parse a tournament seed string like 'W01a' into (region, seed_number).

    Args:
        seed_str: e.g. 'W01', 'X16a', 'Y08b'

    Returns:
        Tuple of (region_letter, seed_int), e.g. ('W', 1)
    """
    m = re.match(r"([WXYZ])(\d{2})", seed_str.strip())
    if not m:
        raise ValueError(f"Cannot parse seed: {seed_str!r}")
    return m.group(1), int(m.group(2))


def seed_int(seed_str: str) -> int:
    """Extract just the numeric seed (1-16) from a seed string."""
    return parse_seed(seed_str)[1]


def build_team_lookup(teams_df) -> dict[int, str]:
    """Build a {TeamID: TeamName} dict from MTeams DataFrame."""
    return dict(zip(teams_df["TeamID"], teams_df["TeamName"]))
