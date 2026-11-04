from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results")
KEY = "career_player_id"

print("=== GOAT REGULAR STATISTICAL PROFILE V1 ===")

df = pd.read_parquet(
    OUT / "unified_goat_player_seasons_v5.parquet"
)

assert not df.duplicated([KEY, "season"]).any()

print("Player-seasons:", len(df))

# Confirm the actual available metric columns.
metrics = [
    "games",
    "minutes",
    "points",
    "assists",
    "rebounds",
    "steals",
    "blocks",
    "per",
    "ws",
    "ws_48",
    "bpm",
    "vorp",
    "ts_percent"
]

available = [
    c for c in metrics
    if c in df.columns
]

print("\nAvailable metrics:", available)

print("\nMissing metrics:", [
    c for c in metrics
    if c not in df.columns
])

# Only aggregate columns that actually exist.
numeric = df[[KEY, "player", "season"] + available].copy()

for col in available:
    numeric[col] = pd.to_numeric(
        numeric[col], errors="coerce"
    )

numeric = numeric.replace(
    [np.inf, -np.inf], np.nan
)

aggregations = {
    "player": "last",
    "season": "nunique"
}

# Additive career totals.
totals = [
    "games", "minutes", "points",
    "assists", "rebounds", "steals",
    "blocks", "ws", "vorp"
]

for col in totals:
    if col in available:
        aggregations[col] = lambda x: x.sum(
            min_count=1
        )

career = (
    numeric.groupby(KEY)
    .agg(aggregations)
    .reset_index()
    .rename(columns={
        "season": "career_seasons"
    })
)

if {"points", "games"}.issubset(career.columns):
    career["career_ppg"] = (
        career["points"]
        / career["games"].replace(0, np.nan)
    )

if {"assists", "games"}.issubset(career.columns):
    career["career_apg"] = (
        career["assists"]
        / career["games"].replace(0, np.nan)
    )

if {"rebounds", "games"}.issubset(career.columns):
    career["career_rpg"] = (
        career["rebounds"]
        / career["games"].replace(0, np.nan)
    )

# Career WS/48 from total WS and total minutes.
if {"ws", "minutes"}.issubset(career.columns):
    career["career_ws48"] = (
        48 * career["ws"]
        / career["minutes"].replace(0, np.nan)
    )

# Peak advanced statistics.
for col in ["per", "bpm", "ws", "vorp"]:
    if col in available:
        peaks = (
            numeric.groupby(KEY)[col]
            .max()
        )

        career["best_season_" + col] = (
            career[KEY].map(peaks)
        )

# Weighted average PER/BPM using minutes.
if "minutes" in available:
    for col in ["per", "bpm"]:
        if col not in available:
            continue

        valid = numeric[
            numeric[col].notna()
            & numeric["minutes"].gt(0)
        ].copy()

        valid["weighted_metric"] = (
            valid[col] * valid["minutes"]
        )

        grouped = valid.groupby(KEY).agg(
            numerator=("weighted_metric", "sum"),
            denominator=("minutes", "sum")
        )

        career["career_weighted_" + col] = (
            career[KEY].map(
                grouped["numerator"]
                / grouped["denominator"]
            )
        )

print("\n=== TOP 20 CAREER WIN SHARES ===")

if "ws" in career.columns:
    print(
        career.nlargest(20, "ws")[
            [KEY, "player", "ws"]
        ].round(2).to_string(index=False)
    )

print("\n=== SELECTED GOAT CANDIDATES ===")

selected = [
    "Michael Jordan",
    "LeBron James",
    "Kareem Abdul-Jabbar",
    "Bill Russell",
    "Wilt Chamberlain",
    "Magic Johnson",
    "Larry Bird",
    "Tim Duncan",
    "Kobe Bryant",
    "Stephen Curry",
    "Nikola Jokić"
]

show = [
    "player",
    "career_seasons",
    "points",
    "career_ppg",
    "assists",
    "career_apg",
    "rebounds",
    "career_rpg",
    "ws",
    "career_ws48",
    "vorp",
    "career_weighted_bpm",
    "best_season_bpm"
]

show = [
    c for c in show
    if c in career.columns
]

print(
    career[
        career["player"].isin(selected)
    ][show]
    .round(2)
    .to_string(index=False)
)

path = OUT / "goat_regular_statistical_profile_v1.csv"

career.to_csv(path, index=False)

print("\nSaved:", path)
print("\nGOAT REGULAR STATISTICAL PROFILE V1 COMPLETE")
