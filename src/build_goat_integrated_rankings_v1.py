from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results")
KEY = "career_player_id"

print("=== GOAT INTEGRATED RANKINGS V1 ===")

sources = {
    "postseason": (
        "goat_postseason_excellence_v3.csv",
        "postseason_excellence_v3"
    ),
    "achievement": (
        "goat_achievement_score_v1.csv",
        "achievement_score_v1"
    ),
    "regular": (
        "goat_regular_excellence_v1.csv",
        "regular_excellence_v1"
    ),
    "peak": (
        "goat_peak_score_v1.csv",
        "peak_dominance_score_v1"
    ),
    "longevity": (
        "goat_longevity_score_v1.csv",
        "longevity_score_v1"
    )
}

weights = {
    "postseason": 0.30,
    "achievement": 0.25,
    "regular": 0.20,
    "peak": 0.20,
    "longevity": 0.05
}

assert np.isclose(sum(weights.values()), 1)

base = pd.read_parquet(
    OUT / "goat_integrated_scoring_base_v1.parquet"
)[
    [KEY, "clean_player"]
].copy()

assert base[KEY].is_unique

score_cols = {}

for name, (filename, col) in sources.items():

    df = pd.read_csv(OUT / filename)

    assert KEY in df.columns
    assert col in df.columns
    assert df[KEY].is_unique

    renamed = f"{name}_score"

    base = base.merge(
        df[[KEY, col]].rename(
            columns={col: renamed}
        ),
        on=KEY,
        how="left",
        validate="one_to_one"
    )

    score_cols[name] = renamed

print("\n=== CATEGORY COVERAGE ===")

for name, col in score_cols.items():

    base[col] = pd.to_numeric(
        base[col], errors="coerce"
    )

    print(
        name,
        "| available:",
        int(base[col].notna().sum())
    )

required = list(score_cols.values())

base["complete_goat_score"] = (
    base[required].notna().all(axis=1)
)

base["goat_score_v1"] = np.nan

complete = base["complete_goat_score"]

base.loc[complete, "goat_score_v1"] = sum(
    weights[name] * base.loc[complete, col]
    for name, col in score_cols.items()
)

base["goat_rank_v1"] = (
    base["goat_score_v1"]
    .rank(ascending=False, method="min")
)

print("\n=== COMPLETE COVERAGE ===")

print("Total players:", len(base))
print(
    "Fully scored players:",
    int(complete.sum())
)

print("\n=== TOP 40 GOAT RANKINGS ===")

show = [
    "clean_player",
    "postseason_score",
    "achievement_score",
    "regular_score",
    "peak_score",
    "longevity_score",
    "goat_score_v1"
]

print(
    base.nlargest(40, "goat_score_v1")[show]
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
    base[
        base["clean_player"].isin(selected)
    ][
        ["goat_rank_v1"] + show
    ]
    .sort_values("goat_rank_v1")
    .round(2)
    .to_string(index=False)
)

path = OUT / "goat_integrated_rankings_v1.csv"

base.to_csv(path, index=False)

print("\nSaved:", path)
print("\nGOAT INTEGRATED RANKINGS V1 COMPLETE")
