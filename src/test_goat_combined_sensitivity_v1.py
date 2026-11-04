from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results")

print("=== GOAT COMBINED SENSITIVITY V1 ===")

df = pd.read_csv(
    OUT / "goat_calibration_sensitivity_v1.csv"
)

weights = pd.read_csv(
    OUT / "goat_weight_sensitivity_draws_v1.csv"
)

categories = [
    "postseason",
    "achievement",
    "regular",
    "peak",
    "longevity"
]

methods = [
    "original",
    "percentile",
    "robust"
]

weight_cols = [
    f"{c}_score" for c in categories
]

W = weights[weight_cols].to_numpy(dtype=float)

assert np.allclose(W.sum(axis=1), 1)

df = df[df["complete_goat_score"]].copy()

assert df["career_player_id"].is_unique

records = []
winners = []

for method in methods:

    if method == "original":
        cols = [
            f"{c}_score" for c in categories
        ]
    else:
        cols = [
            f"{c}_score_{method}"
            for c in categories
        ]

    assert all(c in df.columns for c in cols)

    valid = df.dropna(subset=cols).copy()

    X = valid[cols].to_numpy(dtype=float)

    scores = X @ W.T

    order = np.argsort(
        -scores,
        axis=0,
        kind="stable"
    )

    ranks = np.empty_like(order)

    ranks[
        order,
        np.arange(len(W))[None, :]
    ] = np.arange(1, len(valid) + 1)[:, None]

    win_counts = np.bincount(
        order[0],
        minlength=len(valid)
    )

    for i, (_, player) in enumerate(
        valid.iterrows()
    ):

        records.append({
            "method": method,
            "career_player_id": player["career_player_id"],
            "clean_player": player["clean_player"],
            "wins": int(win_counts[i]),
            "win_rate": win_counts[i] / len(W),
            "mean_rank": ranks[i].mean(),
            "median_rank": np.median(ranks[i]),
            "best_rank": ranks[i].min(),
            "worst_rank": ranks[i].max()
        })

    winner_names = valid.iloc[
        order[0]
    ]["clean_player"].to_numpy()

    for name in winner_names:
        winners.append({
            "method": method,
            "winner": name
        })

summary = pd.DataFrame(records)

print("\n=== TOP 10 BY CALIBRATION ===")

for method in methods:

    print("\nMETHOD:", method)

    sub = summary[
        summary["method"].eq(method)
    ].sort_values(
        ["win_rate", "mean_rank"],
        ascending=[False, True]
    )

    print(
        sub.head(10)
        .round(3)
        .to_string(index=False)
    )

print("\n=== OVERALL WIN FREQUENCY ===")

winner_df = pd.DataFrame(winners)

overall = (
    winner_df["winner"]
    .value_counts()
    .rename_axis("clean_player")
    .reset_index(name="wins")
)

overall["win_rate"] = (
    overall["wins"] / len(winner_df)
)

print(
    overall.head(20)
    .round(4)
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
            "best_rank",
            "worst_rank"
        ]
    ]
    .round(4)
    .to_string(index=False)
)

summary.to_csv(
    OUT / "goat_combined_sensitivity_v1.csv",
    index=False
)

overall.to_csv(
    OUT / "goat_combined_sensitivity_winners_v1.csv",
    index=False
)

print("\nGOAT COMBINED SENSITIVITY V1 COMPLETE")
