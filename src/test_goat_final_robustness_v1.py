from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results")
RNG = np.random.default_rng(20261019)

print("=== GOAT FINAL ROBUSTNESS V1 ===")

df = pd.read_csv(
    OUT / "goat_reference_threshold_stability_v1.csv"
)

categories = [
    "postseason",
    "achievement",
    "regular",
    "peak",
    "longevity"
]

base_weights = np.array([
    0.30, 0.25, 0.20, 0.20, 0.05
])

original_cols = [
    "postseason_score",
    "achievement_fractional_v2",
    "regular_score",
    "peak_score",
    "longevity_score"
]

methods = [
    "original",
    "minutes_25000",
    "minutes_30000",
    "minutes_35000"
]

complete = df[original_cols].notna().all(axis=1)

valid = df.loc[complete].copy().reset_index(drop=True)

print("Fully scored players:", len(valid))

assert len(valid) > 100

# Build each method's category matrix.
matrices = {}

matrices["original"] = (
    valid[original_cols].to_numpy(dtype=float)
)

for method in methods[1:]:

    threshold = int(method.split("_")[1])

    reference = df[
        df["regular_minutes"].ge(threshold)
    ]

    columns = []

    for col in original_cols:

        values = pd.to_numeric(
            reference[col],
            errors="coerce"
        ).dropna()

        q25 = values.quantile(0.25)
        q95 = values.quantile(0.95)

        spread = q95 - q25

        assert spread > 0

        calibrated = (
            50
            + 25 * (valid[col] - q25) / spread
        )

        columns.append(
            calibrated.to_numpy(dtype=float)
        )

    matrices[method] = np.column_stack(columns)

# Weight uncertainty:
# Dirichlet draws centered on preferred weights.
#
# Concentration 100 creates moderate variation
# while preserving the average intended weights.

N = 10000

weights = RNG.dirichlet(
    base_weights * 100,
    size=N
)

records = []
winner_records = []

for method, X in matrices.items():

    scores = X @ weights.T

    order = np.argsort(
        -scores,
        axis=0,
        kind="stable"
    )

    ranks = np.empty_like(order)

    ranks[
        order,
        np.arange(N)[None, :]
    ] = np.arange(1, len(valid) + 1)[:, None]

    winners = order[0, :]

    counts = np.bincount(
        winners,
        minlength=len(valid)
    )

    for i, player in valid.iterrows():

        records.append({
            "method": method,
            "career_player_id": player["career_player_id"],
            "clean_player": player["clean_player"],
            "win_count": int(counts[i]),
            "win_rate": counts[i] / N,
            "mean_rank": ranks[i].mean(),
            "median_rank": np.median(ranks[i]),
            "top5_rate": (ranks[i] <= 5).mean(),
            "best_rank": ranks[i].min(),
            "worst_rank": ranks[i].max()
        })

    for i in winners:
        winner_records.append({
            "method": method,
            "winner": valid.iloc[i]["clean_player"]
        })

summary = pd.DataFrame(records)

print("\n=== TOP 12 BY METHOD ===")

for method in methods:

    print("\nMETHOD:", method)

    sub = summary[
        summary["method"].eq(method)
    ].sort_values(
        ["win_rate", "mean_rank"],
        ascending=[False, True]
    )

    print(
        sub.head(12)[
            [
                "clean_player",
                "win_rate",
                "mean_rank",
                "top5_rate",
                "best_rank",
                "worst_rank"
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

print("\n=== JORDAN VS LEBRON ===")

selected = summary[
    summary["clean_player"].isin([
        "Michael Jordan",
        "LeBron James"
    ])
]

print(
    selected[
        [
            "method",
            "clean_player",
            "win_rate",
            "mean_rank",
            "top5_rate"
        ]
    ]
    .round(4)
    .to_string(index=False)
)

summary.to_csv(
    OUT / "goat_final_robustness_v1.csv",
    index=False
)

pd.DataFrame(winner_records).to_csv(
    OUT / "goat_final_robustness_winners_v1.csv",
    index=False
)

print("\nGOAT FINAL ROBUSTNESS V1 COMPLETE")
