from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results")

print("=== GOAT ACHIEVEMENT SCORE V1 ===")

df = pd.read_csv(
    OUT / "goat_achievement_inputs_v1.csv"
)

# Component weights
WEIGHTS = {
    "championship": 0.45,
    "mvp": 0.25,
    "finals_mvp": 0.15,
    "other": 0.15
}

# Fixed reference anchors for reproducibility.
# These are provisional and will be sensitivity-tested.
ANCHORS = {
    "championship": 2.5,
    "mvp": 6.0,
    "finals_mvp": 6.0,
    "all_nba": 21.0,
    "all_def": 15.0,
    "dpoy": 4.0
}

def score(value, anchor):
    x = pd.to_numeric(
        value, errors="coerce"
    ).clip(lower=0)

    return 100 * np.arcsinh(x) / np.arcsinh(anchor)

df["championship_score"] = score(
    df["championship_credit_a10"],
    ANCHORS["championship"]
)

df["mvp_score"] = score(
    df["mvp_wins_v2"],
    ANCHORS["mvp"]
)

df["finals_mvp_score"] = score(
    df["finals_mvp"],
    ANCHORS["finals_mvp"]
)

df["other_score"] = (
    0.55 * score(
        df["all_nba_total"],
        ANCHORS["all_nba"]
    )
    + 0.25 * score(
        df["all_def_total"],
        ANCHORS["all_def"]
    )
    + 0.20 * score(
        df["dpoy_wins"],
        ANCHORS["dpoy"]
    )
)

# Availability flags reflect whether the award
# existed during any portion of the player's career.
df["finals_mvp_available"] = (
    df["finals_mvp_era_seasons"].gt(0)
)

df["dpoy_available"] = (
    df["dpoy_eligible_seasons"].gt(0)
)

df["all_def_available"] = (
    df["all_def_eligible_seasons"].gt(0)
)

# Calculate historically available honors,
# redistributing unavailable subcomponent weight.
other_parts = {
    "all_nba_total": 0.55,
    "all_def_total": 0.25,
    "dpoy_wins": 0.20
}

other_scores = pd.DataFrame({
    "all_nba_total": score(
        df["all_nba_total"], ANCHORS["all_nba"]
    ),
    "all_def_total": score(
        df["all_def_total"], ANCHORS["all_def"]
    ),
    "dpoy_wins": score(
        df["dpoy_wins"], ANCHORS["dpoy"]
    )
})

other_available = pd.DataFrame({
    "all_nba_total": pd.Series(
        True, index=df.index
    ),
    "all_def_total": df["all_def_available"],
    "dpoy_wins": df["dpoy_available"]
})

other_weights = pd.Series(other_parts)

numerator = (
    other_scores
    .mul(other_weights, axis=1)
    .where(other_available)
    .sum(axis=1)
)

denominator = (
    other_available
    .mul(other_weights, axis=1)
    .sum(axis=1)
)

df["other_score"] = (
    numerator / denominator.replace(0, np.nan)
)

components = {
    "championship_score": WEIGHTS["championship"],
    "mvp_score": WEIGHTS["mvp"],
    "finals_mvp_score": WEIGHTS["finals_mvp"],
    "other_score": WEIGHTS["other"]
}

available = pd.DataFrame({
    "championship_score":
        df["championship_contribution_known"],
    "mvp_score":
        df["mvp_score"].notna(),
    "finals_mvp_score":
        df["finals_mvp_available"],
    "other_score":
        df["other_score"].notna()
})

weighted_sum = pd.Series(
    0.0, index=df.index
)

weight_sum = pd.Series(
    0.0, index=df.index
)

for col, weight in components.items():

    valid = available[col] & df[col].notna()

    weighted_sum += (
        df[col].fillna(0) * weight * valid
    )

    weight_sum += weight * valid

df["achievement_score_v1"] = (
    weighted_sum / weight_sum.replace(0, np.nan)
)

# Require verified championship contribution
# for a complete achievement score.
df.loc[
    ~df["championship_contribution_known"],
    "achievement_score_v1"
] = np.nan

print("\n=== COVERAGE ===")
print("Players:", len(df))
print(
    "Complete achievement scores:",
    df["achievement_score_v1"].notna().sum()
)

print("\n=== TOP 30 ACHIEVEMENT SCORES ===")

show = [
    "clean_player",
    "championship_score",
    "mvp_score",
    "finals_mvp_score",
    "other_score",
    "achievement_score_v1"
]

print(
    df.nlargest(
        30, "achievement_score_v1"
    )[show]
    .round(2)
    .to_string(index=False)
)

print("\n=== SELECTED GOAT CANDIDATES ===")

selected = [
    "Michael Jordan",
    "LeBron James",
    "Bill Russell",
    "Wilt Chamberlain",
    "Kareem Abdul-Jabbar",
    "Magic Johnson",
    "Larry Bird",
    "Tim Duncan",
    "Kobe Bryant",
    "Stephen Curry",
    "Shaquille O'Neal",
    "Hakeem Olajuwon",
    "Nikola Jokić"
]

print(
    df[df["clean_player"].isin(selected)][show]
    .round(2)
    .to_string(index=False)
)

path = OUT / "goat_achievement_score_v1.csv"

df.to_csv(path, index=False)

print("\nSaved:", path)
print("\nGOAT ACHIEVEMENT SCORE V1 COMPLETE")
