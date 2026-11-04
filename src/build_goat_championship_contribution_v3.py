from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results")
TEAM = ["season", "team_key"]

print("=== GOAT CHAMPIONSHIP CONTRIBUTION V3 ===")

df = pd.read_csv(
    OUT / "goat_championship_coverage_recovery_v1.csv"
)

assert df["playoff_record_id"].is_unique

df = df[
    df["season_ending_year"].lt(2026)
].copy()

for col in [
    "postseason_possessions",
    "postseason_minutes",
    "total_impact_prior",
    "playoff_ws48"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Only teams with complete recoverable evidence
# can receive measured contribution shares.
team_ok = (
    df.groupby(TEAM)["recovery_eligible"]
    .transform("all")
)

df["complete_v3_roster"] = team_ok

def roster_zscore(series):
    mean = series.mean()
    std = series.std(ddof=0)

    if pd.isna(std) or std < 1e-10:
        return pd.Series(0.0, index=series.index)

    return (series - mean) / std

# Standardize playoff performance among players
# who actually participated in the postseason.
active = df["postseason_minutes"].gt(0)

df["regular_z_v3"] = np.nan
df["playoff_z_v3"] = np.nan

df.loc[active, "regular_z_v3"] = (
    df[active]
    .groupby(TEAM)["total_impact_prior"]
    .transform(roster_zscore)
)

df.loc[active, "playoff_z_v3"] = (
    df[active]
    .groupby(TEAM)["playoff_ws48"]
    .transform(roster_zscore)
)

# Missing regular-season impact gets a neutral
# prior, with playoff performance carrying
# the entire weight for those specific players.
post_only = df["recovery_status"].eq("POSTSEASON_ONLY")

df.loc[post_only, "regular_z_v3"] = 0.0

df["blended_z_v3"] = (
    0.60 * df["playoff_z_v3"]
    + 0.40 * df["regular_z_v3"]
)

df.loc[post_only, "blended_z_v3"] = (
    df.loc[post_only, "playoff_z_v3"]
)

# Soft exponential allocation.
df["intensity_v3"] = np.exp(
    0.5 * df["blended_z_v3"].clip(-4, 4)
)

df["value_v3"] = (
    df["postseason_possessions"]
    * df["intensity_v3"]
)

# Zero-minute players have no measured
# contribution to the playoff run.
zero = df["recovery_status"].eq("ZERO_MINUTES")

df.loc[zero, "value_v3"] = 0.0

df.loc[
    ~df["complete_v3_roster"],
    "value_v3"
] = np.nan

team_total = (
    df.groupby(TEAM)["value_v3"]
    .transform("sum")
)

df["contribution_share_v3"] = (
    df["value_v3"]
    / team_total.replace(0, np.nan)
)

df.loc[
    ~df["complete_v3_roster"],
    "contribution_share_v3"
] = np.nan

complete = df[df["complete_v3_roster"]]

shares = (
    complete.groupby(TEAM)["contribution_share_v3"]
    .sum()
)

assert np.allclose(shares, 1.0)

print("\n=== COVERAGE ===")
print("Championship teams:", df.groupby(TEAM).ngroups)
print(
    "Complete V3 teams:",
    complete.groupby(TEAM).ngroups
)

print("\n=== TOP CAREER CONTRIBUTIONS ===")

career = (
    df.groupby("career_player_id")
    .agg(
        player=("Player", "last"),
        championships=("season", "nunique"),
        measured_championships=(
            "contribution_share_v3", "count"
        ),
        total_contribution_v3=(
            "contribution_share_v3",
            lambda x: x.sum(min_count=1)
        )
    )
    .reset_index()
)

print(
    career.nlargest(
        25, "total_contribution_v3"
    )
    .round(3)
    .to_string(index=False)
)

print("\n=== SELECTED GOAT CANDIDATES ===")

selected = [
    "Michael Jordan",
    "LeBron James",
    "Bill Russell",
    "Kareem Abdul-Jabbar",
    "Magic Johnson",
    "Tim Duncan",
    "Kobe Bryant",
    "Stephen Curry",
    "Shaquille O'Neal",
    "Hakeem Olajuwon"
]

career["clean_player"] = (
    career["player"]
    .str.replace("*", "", regex=False)
)

print(
    career[
        career["clean_player"].isin(selected)
    ]
    .sort_values(
        "total_contribution_v3",
        ascending=False
    )
    .round(3)
    .to_string(index=False)
)

season_path = (
    OUT / "goat_championship_contribution_seasons_v3.csv"
)

career_path = (
    OUT / "goat_championship_contribution_career_v3.csv"
)

df.to_csv(season_path, index=False)
career.to_csv(career_path, index=False)

print("\nSaved:", season_path)
print("Saved:", career_path)

print("\nGOAT CHAMPIONSHIP CONTRIBUTION V3 COMPLETE")
