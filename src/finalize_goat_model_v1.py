from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

OUT = Path("results")
OUT.mkdir(exist_ok=True)

KEY = "career_player_id"

print("=== GOAT FINAL MODEL VALIDATION V1 ===")

source = OUT / "goat_reference_threshold_stability_v1.csv"

df = pd.read_csv(source)

assert df[KEY].is_unique
assert df["clean_player"].notna().all()

categories = {
    "postseason": {
        "column": "postseason_score",
        "weight": 0.30
    },
    "achievement": {
        "column": "achievement_fractional_v2",
        "weight": 0.25
    },
    "regular": {
        "column": "regular_score",
        "weight": 0.20
    },
    "peak": {
        "column": "peak_score",
        "weight": 0.20
    },
    "longevity": {
        "column": "longevity_score",
        "weight": 0.05
    }
}

assert np.isclose(
    sum(x["weight"] for x in categories.values()),
    1.0
)

threshold = 30000

required = [
    config["column"]
    for config in categories.values()
]

for col in required + ["regular_minutes"]:
    assert col in df.columns
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

complete = df[required].notna().all(axis=1)

reference = df[
    df["regular_minutes"].ge(threshold)
].copy()

print("\n=== POPULATION ===")
print("Total players:", len(df))
print("Reference players:", len(reference))
print("Fully scored:", int(complete.sum()))

assert len(reference) >= 100
assert complete.sum() >= 1000

model = {
    "model_name": "NBA GOAT Model",
    "version": "Final Candidate V1",
    "reference_minimum_minutes": threshold,
    "calibration": (
        "50 + 25 * (raw_score - Q25) / (Q95 - Q25)"
    ),
    "historical_awards": "fractional eligibility",
    "categories": categories,
    "reference_quantiles": {}
}

score = pd.Series(0.0, index=df.index)

for category, config in categories.items():

    col = config["column"]
    weight = config["weight"]

    values = reference[col].dropna()

    q25 = float(values.quantile(.25))
    q95 = float(values.quantile(.95))

    spread = q95 - q25

    assert np.isfinite(spread) and spread > 0

    calibrated = (
        50 + 25 * (df[col] - q25) / spread
    )

    df[f"final_{category}_calibrated"] = calibrated

    df[f"final_{category}_contribution"] = (
        calibrated * weight
    )

    score += (
        df[f"final_{category}_contribution"]
        .fillna(0)
    )

    model["reference_quantiles"][category] = {
        "q25": q25,
        "q95": q95,
        "spread": spread
    }

df["final_goat_score"] = score.where(complete)

df["final_goat_rank"] = (
    df["final_goat_score"]
    .rank(ascending=False, method="min")
)

print("\n=== REPRODUCIBILITY CHECK ===")

previous_col = "goat_score_minutes_30000"

assert previous_col in df.columns

valid = df[
    df["final_goat_score"].notna()
    & df[previous_col].notna()
]

difference = (
    valid["final_goat_score"]
    - valid[previous_col]
).abs()

print("Compared:", len(valid))
print("Maximum absolute difference:", difference.max())

assert len(valid) == int(complete.sum())
assert difference.max() < 1e-6

# Check category contribution arithmetic.
contribution_cols = [
    f"final_{category}_contribution"
    for category in categories
]

reconstructed = df.loc[
    complete, contribution_cols
].sum(axis=1)

assert np.allclose(
    reconstructed,
    df.loc[complete, "final_goat_score"]
)

print("\n=== FINAL TOP 30 ===")

show = [
    "final_goat_rank",
    "clean_player",
    "final_goat_score"
]

print(
    df.nsmallest(
        30, "final_goat_rank"
    )[show]
    .round(4)
    .to_string(index=False)
)

print("\n=== SCORE RANGE CHECK ===")

print(
    df.loc[complete, "final_goat_score"]
    .describe()
    .round(4)
    .to_string()
)

print("\n=== HISTORICAL CANDIDATE COVERAGE ===")

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
    "Shaquille O'Neal",
    "Hakeem Olajuwon",
    "Nikola Jokić"
]

candidates = df[
    df["clean_player"].isin(selected)
]

assert len(candidates) == len(selected)
assert candidates["final_goat_score"].notna().all()

print(
    candidates[show]
    .sort_values("final_goat_rank")
    .round(4)
    .to_string(index=False)
)

# Save reproducible outputs.
leaderboard = df[
    df["final_goat_score"].notna()
].sort_values(
    ["final_goat_score", KEY],
    ascending=[False, True]
).copy()

leaderboard.to_csv(
    OUT / "goat_final_leaderboard_v1.csv",
    index=False
)

leaderboard.head(100).to_csv(
    OUT / "goat_final_top100_v1.csv",
    index=False
)

with open(
    OUT / "goat_final_model_specification_v1.json",
    "w"
) as f:
    json.dump(model, f, indent=2)

# Record SHA256 hash of source data.
sha256 = hashlib.sha256(
    source.read_bytes()
).hexdigest()

report = {
    "source": str(source),
    "source_sha256": sha256,
    "total_players": len(df),
    "reference_players": len(reference),
    "fully_scored_players": int(complete.sum()),
    "maximum_reconstruction_error": float(
        difference.max()
    ),
    "status": "REPRODUCIBILITY CHECK PASSED"
}

with open(
    OUT / "goat_final_validation_report_v1.json",
    "w"
) as f:
    json.dump(report, f, indent=2)

print("\n=== SAVED FILES ===")
print("goat_final_leaderboard_v1.csv")
print("goat_final_top100_v1.csv")
print("goat_final_model_specification_v1.json")
print("goat_final_validation_report_v1.json")

print("\nGOAT FINAL MODEL VALIDATION COMPLETE")
