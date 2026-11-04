from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results")

print("=== GOAT PEAK DOMINANCE SCORE V1 ===")

df = pd.read_csv(
    OUT / "goat_peak_dominance_v2.csv"
).copy()

KEY = "career_player_id"

assert df[KEY].is_unique

metrics = {
    "single": "best1_value",
    "three": "best3_consecutive_value",
    "five": "best5_consecutive_value"
}

for col in metrics.values():
    assert col in df.columns
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

# Reference players must have complete
# one-, three-, and five-year peak evidence.
complete = df[
    list(metrics.values())
].notna().all(axis=1)

reference = df[complete].copy()

print("\n=== COVERAGE ===")
print("Players:", len(df))
print("Complete peak records:", int(complete.sum()))

# Use a log-like transform that handles
# zero and negative values safely.
def transform_score(values, reference_values):

    ref = reference_values.dropna()

    scale = np.median(np.abs(ref))

    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    transformed_ref = np.arcsinh(ref / scale)
    transformed = np.arcsinh(values / scale)

    # Center on the reference distribution.
    center = np.median(transformed_ref)

    q25, q75 = np.percentile(
        transformed_ref, [25, 75]
    )

    spread = q75 - q25

    if spread <= 1e-10:
        spread = np.std(transformed_ref)

    spread = max(spread, 1e-10)

    z = (transformed - center) / spread

    return 100 / (
        1 + np.exp(-0.8 * z.clip(-20, 20))
    )

for name, col in metrics.items():

    df[f"{name}_peak_score"] = transform_score(
        df[col],
        reference[col]
    )

df["peak_dominance_score_v1"] = (
    0.25 * df["single_peak_score"]
    + 0.30 * df["three_peak_score"]
    + 0.45 * df["five_peak_score"]
)

df.loc[
    ~complete,
    "peak_dominance_score_v1"
] = np.nan

print("\n=== TOP 30 PEAK DOMINANCE ===")

show = [
    KEY,
    "single_peak_score",
    "three_peak_score",
    "five_peak_score",
    "peak_dominance_score_v1"
]

print(
    df.nlargest(
        30, "peak_dominance_score_v1"
    )[show]
    .round(2)
    .to_string(index=False)
)

# Add names from verified integrated base.
names = pd.read_parquet(
    OUT / "goat_integrated_scoring_base_v1.parquet"
)[
    [KEY, "clean_player"]
].drop_duplicates()

assert names[KEY].is_unique

df = df.merge(
    names,
    on=KEY,
    how="left",
    validate="one_to_one"
)

print("\n=== TOP 30 WITH NAMES ===")

print(
    df.nlargest(
        30, "peak_dominance_score_v1"
    )[
        [
            "clean_player",
            "single_peak_score",
            "three_peak_score",
            "five_peak_score",
            "peak_dominance_score_v1"
        ]
    ]
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
    df[
        df["clean_player"].isin(selected)
    ][
        [
            "clean_player",
            "single_peak_score",
            "three_peak_score",
            "five_peak_score",
            "peak_dominance_score_v1"
        ]
    ]
    .round(2)
    .to_string(index=False)
)

path = OUT / "goat_peak_score_v1.csv"

df.to_csv(path, index=False)

print("\nSaved:", path)
print("\nGOAT PEAK DOMINANCE SCORE V1 COMPLETE")
