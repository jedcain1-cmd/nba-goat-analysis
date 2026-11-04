from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results")

print("=== GOAT POSTSEASON EXCELLENCE V1 ===")

df = pd.read_parquet(
    OUT / "goat_postseason_scoring_base_v1.parquet"
).copy()

KEY = "career_player_id"

assert df[KEY].is_unique

metrics = {
    "career_value": "postseason_value",
    "peak_value": "best5_consecutive_value",
    "efficiency": "weighted_playoff_ts_above_era"
}

print("\n=== AVAILABLE EFFICIENCY COLUMNS ===")

print([
    c for c in df.columns
    if "ts_above" in c.lower()
    or "efficiency" in c.lower()
])

# Resolve expected column names.
resolved = {}

for name, expected in metrics.items():

    candidates = [
        c for c in df.columns
        if c == expected
        or c.endswith("_" + expected)
    ]

    if candidates:
        resolved[name] = candidates[0]

print("\n=== RESOLVED METRICS ===")
print(resolved)

assert len(resolved) == len(metrics), (
    "Required postseason scoring metrics missing"
)

for name, col in resolved.items():
    df[name] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Reference population: players with positive
# career playoff value and a measured peak.
reference = df[
    df["career_value"].gt(0)
    & df["peak_value"].notna()
    & df["efficiency"].notna()
].copy()

print("\nReference players:", len(reference))

def percentile_score(values, reference_values):
    reference_values = (
        reference_values.dropna().sort_values()
    )

    if reference_values.empty:
        return pd.Series(np.nan, index=values.index)

    ranks = np.searchsorted(
        reference_values.to_numpy(),
        values.to_numpy(),
        side="right"
    )

    result = 100 * ranks / len(reference_values)

    return pd.Series(
        result,
        index=values.index
    ).where(values.notna())

for name in metrics:

    df[f"{name}_score"] = percentile_score(
        df[name],
        reference[name]
    )

weights = {
    "career_value_score": 0.50,
    "peak_value_score": 0.30,
    "efficiency_score": 0.20
}

complete = df[
    list(weights)
].notna().all(axis=1)

df["postseason_excellence_v1"] = np.nan

df.loc[
    complete,
    "postseason_excellence_v1"
] = sum(
    df.loc[complete, col] * weight
    for col, weight in weights.items()
)

df["postseason_score_complete"] = complete

print("\n=== COVERAGE ===")
print("Players:", len(df))
print("Complete scores:", int(complete.sum()))

print("\n=== TOP 30 POSTSEASON EXCELLENCE ===")

name_col = next(
    c for c in [
        "clean_player", "player", "Player"
    ]
    if c in df.columns
)

show = [
    name_col,
    "career_value_score",
    "peak_value_score",
    "efficiency_score",
    "postseason_excellence_v1"
]

print(
    df.nlargest(
        30, "postseason_excellence_v1"
    )[show]
    .round(2)
    .to_string(index=False)
)

path = OUT / "goat_postseason_excellence_v1.csv"

df.to_csv(path, index=False)

print("\nSaved:", path)
print("\nGOAT POSTSEASON EXCELLENCE V1 COMPLETE")
