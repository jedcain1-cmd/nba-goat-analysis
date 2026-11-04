from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results")
KEY = "career_player_id"

print("=== GOAT POSTSEASON PEAK V1 ===")

post = pd.read_parquet(
    OUT / "goat_postseason_value_baseline_v2.parquet"
)

regular = pd.read_parquet(
    OUT / "goat_regular_season_value_v1.parquet"
)

required_post = [
    KEY,
    "season_ending_year",
    "postseason_value_baseline",
    "valuation_eligible"
]

required_regular = [
    KEY,
    "season_ending_year"
]

assert all(c in post.columns for c in required_post)
assert all(c in regular.columns for c in required_regular)

post = post[required_post].copy()

post["season_ending_year"] = pd.to_numeric(
    post["season_ending_year"],
    errors="coerce"
)

post["postseason_value_baseline"] = pd.to_numeric(
    post["postseason_value_baseline"],
    errors="coerce"
)

assert not post.duplicated(
    [KEY, "season_ending_year"]
).any()

# Regular-season records establish which
# NBA seasons the player actually participated in.
career_years = regular[
    required_regular
].drop_duplicates().copy()

career_years["season_ending_year"] = pd.to_numeric(
    career_years["season_ending_year"],
    errors="coerce"
)

assert not career_years.duplicated(
    [KEY, "season_ending_year"]
).any()

timeline = career_years.merge(
    post,
    on=[KEY, "season_ending_year"],
    how="left",
    validate="one_to_one",
    indicator=True
)

# Missing postseason rows are only treated as
# zero when the regular-season record verifies
# that the player participated that year.
#
# This is a provisional assumption and should
# be audited against participation flags.
timeline["peak_value"] = np.where(
    timeline["_merge"].eq("left_only"),
    0.0,
    np.where(
        timeline["valuation_eligible"].eq(True),
        timeline["postseason_value_baseline"],
        np.nan
    )
)

timeline["evidence_status"] = np.select(
    [
        timeline["_merge"].eq("left_only"),
        timeline["valuation_eligible"].eq(True)
    ],
    [
        "NO_POSTSEASON_RECORD",
        "MEASURED_POSTSEASON"
    ],
    default="UNVERIFIED"
)

print("\n=== TIMELINE COVERAGE ===")
print("Player-seasons:", len(timeline))
print(
    timeline["evidence_status"]
    .value_counts()
    .to_string()
)

records = []

for player_id, group in timeline.groupby(KEY):

    group = group.sort_values("season_ending_year")

    years = group["season_ending_year"].astype(int).to_numpy()
    values = group["peak_value"].to_numpy(dtype=float)

    result = {KEY: player_id}

    for window in [3, 5, 7]:

        best = np.nan
        best_start = np.nan

        for start in range(
            int(years.min()),
            int(years.max()) - window + 2
        ):

            target_years = np.arange(
                start, start + window
            )

            subset = group[
                group["season_ending_year"].isin(
                    target_years
                )
            ]

            if len(subset) != window:
                continue

            vals = subset["peak_value"]

            if vals.isna().any():
                continue

            total = vals.sum()

            if pd.isna(best) or total > best:
                best = total
                best_start = start

        result[f"postseason_best{window}_value"] = best
        result[f"postseason_best{window}_start"] = best_start

    records.append(result)

peak = pd.DataFrame(records)

print("\n=== PEAK COVERAGE ===")

for window in [3, 5, 7]:
    col = f"postseason_best{window}_value"

    print(
        col,
        "| available:",
        peak[col].notna().sum()
    )

print("\n=== TOP FIVE-YEAR PLAYOFF PEAKS ===")

print(
    peak.nlargest(
        25, "postseason_best5_value"
    )
    .round(3)
    .to_string(index=False)
)

path = OUT / "goat_postseason_peak_v1.csv"

peak.to_csv(path, index=False)

print("\nSaved:", path)
print("\nGOAT POSTSEASON PEAK V1 COMPLETE")
