from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

OUT = Path("results")
OUT.mkdir(exist_ok=True)

SOURCE = OUT / "goat_final_candidate_comparison_v1.csv"
KEY = "career_player_id"

df = pd.read_csv(SOURCE)

assert df[KEY].is_unique
assert df["clean_player"].notna().all()

weights = {
    "postseason_score": 0.30,
    "achievement_fractional_v2": 0.25,
    "regular_score": 0.20,
    "peak_score": 0.20,
    "longevity_score": 0.05
}

assert np.isclose(sum(weights.values()), 1)

score_col = "goat_score_fractional_v2"

required = [KEY, "clean_player", score_col] + list(weights)

missing = [c for c in required if c not in df.columns]
assert not missing, f"Missing columns: {missing}"

# Reconstruct the final score independently.
components = pd.DataFrame({
    name: pd.to_numeric(df[name], errors="coerce") * weight
    for name, weight in weights.items()
})

complete = components.notna().all(axis=1)

reconstructed = components.sum(axis=1, min_count=len(weights))

original = pd.to_numeric(df[score_col], errors="coerce")

valid = complete & original.notna()

assert valid.sum() > 1000

difference = (
    reconstructed.loc[valid]
    - original.loc[valid]
).abs()

maximum_error = float(difference.max())

assert maximum_error < 1e-6, (
    f"Score reconstruction failed: {maximum_error}"
)

# Verify that incomplete rows are not scored.
assert not (original.notna() & ~complete).any()

leaderboard = df[[KEY, "clean_player"]].copy()

for col in weights:
    leaderboard[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

leaderboard["final_goat_score"] = original

leaderboard["final_goat_rank"] = (
    leaderboard["final_goat_score"]
    .rank(ascending=False, method="min")
)

leaderboard = leaderboard.sort_values(
    ["final_goat_rank", KEY],
    na_position="last"
).reset_index(drop=True)

# Save frozen rankings.
leaderboard.to_csv(
    OUT / "goat_official_leaderboard_v1.csv",
    index=False
)

leaderboard.head(100).to_csv(
    OUT / "goat_official_top100_v1.csv",
    index=False
)

# Source checksum.
sha256 = hashlib.sha256(
    SOURCE.read_bytes()
).hexdigest()

specification = {
    "model_name": "NBA GOAT - Corrected V1",
    "status": "FROZEN",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "source_file": str(SOURCE),
    "source_sha256": sha256,
    "score_column": score_col,
    "category_weights": weights,
    "historical_award_treatment": "fractional eligibility",
    "scope": "NBA careers",
    "notes": [
        "Measures overall NBA greatness, including performance and achievements.",
        "Uses original V1 category scoring transformations.",
        "Historical award availability uses fractional eligibility.",
        "V2 elite-reference calibration retained as sensitivity analysis.",
        "Rankings are model-dependent, not objective ground truth."
    ]
}

with open(
    OUT / "goat_official_specification_v1.json", "w"
) as f:
    json.dump(specification, f, indent=2)

report = {
    "total_players": int(len(df)),
    "fully_scored_players": int(valid.sum()),
    "maximum_reconstruction_error": maximum_error,
    "source_sha256": sha256,
    "validation_passed": True
}

with open(
    OUT / "goat_official_validation_v1.json", "w"
) as f:
    json.dump(report, f, indent=2)

print("=== OFFICIAL NBA GOAT MODEL ===")
print("Total players:", len(df))
print("Fully scored:", int(valid.sum()))
print("Maximum score error:", maximum_error)

print("\n=== OFFICIAL TOP 30 ===")

print(
    leaderboard.head(30)[
        [
            "final_goat_rank",
            "clean_player",
            "final_goat_score"
        ]
    ]
    .round(3)
    .to_string(index=False)
)

print("\n=== SAVED FILES ===")

for filename in [
    "goat_official_leaderboard_v1.csv",
    "goat_official_top100_v1.csv",
    "goat_official_specification_v1.json",
    "goat_official_validation_v1.json"
]:
    print(OUT / filename)

print("\nGOAT CORRECTED V1 FREEZE COMPLETE")
