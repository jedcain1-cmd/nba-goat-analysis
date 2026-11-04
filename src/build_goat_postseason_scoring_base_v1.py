from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("results")
KEY = "career_player_id"

print("=== GOAT POSTSEASON SCORING BASE V1 ===")

sources = {
    "evidence": (
        "goat_postseason_evidence_v1.csv"
    ),
    "statistics": (
        "goat_postseason_statistical_profile_v2.csv"
    ),
    "advanced": (
        "goat_playoff_advanced_career_v2.csv"
    ),
    "peak": (
        "goat_peak_dominance_v2.csv"
    )
}

frames = {}

for name, filename in sources.items():

    df = pd.read_csv(OUT / filename)

    assert KEY in df.columns, (
        f"{name}: missing player ID"
    )

    assert not df[KEY].duplicated().any(), (
        f"{name}: duplicate player IDs"
    )

    frames[name] = df

    print(
        name,
        "| records:",
        len(df),
        "| columns:",
        len(df.columns)
    )

# Begin with all players in the postseason
# evidence database.
base = frames["evidence"].copy()

for name in [
    "statistics",
    "advanced",
    "peak"
]:

    other = frames[name].copy()

    # Preserve the canonical name from evidence.
    # Prefix other overlapping columns to prevent
    # accidental overwriting.
    overlaps = (
        set(base.columns)
        & set(other.columns)
    ) - {KEY}

    other = other.rename(columns={
        c: f"{name}_{c}"
        for c in overlaps
    })

    base = base.merge(
        other,
        on=KEY,
        how="left",
        validate="one_to_one"
    )

assert len(base) == len(frames["evidence"])
assert not base[KEY].duplicated().any()

print("\n=== INTEGRATED COVERAGE ===")

print("Players:", len(base))
print("Columns:", len(base.columns))

metrics = [
    "postseason_value",
    "career_pts",
    "playoff_ws",
    "playoff_vorp",
    "weighted_playoff_bpm",
    "career_playoff_ts",
    "best5_consecutive_value",
    "best7_consecutive_value"
]

print("\n=== METRIC AVAILABILITY ===")

for metric in metrics:

    matches = [
        c for c in base.columns
        if c == metric
        or c.endswith("_" + metric)
    ]

    for col in matches:

        values = pd.to_numeric(
            base[col],
            errors="coerce"
        )

        print(
            col,
            "| available:",
            values.notna().sum()
        )

print("\n=== SELECTED GOAT CANDIDATES ===")

name_col = next(
    (
        c for c in [
            "clean_player",
            "player",
            "Player"
        ]
        if c in base.columns
    ),
    None
)

if name_col:

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
        "Nikola Jokić"
    ]

    clean = (
        base[name_col]
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    show = [KEY, name_col]

    for metric in metrics:
        matches = [
            c for c in base.columns
            if c == metric
            or c.endswith("_" + metric)
        ]

        if matches:
            show.append(matches[0])

    print(
        base.loc[clean.isin(selected), show]
        .round(3)
        .to_string(index=False)
    )

path = (
    OUT / "goat_postseason_scoring_base_v1.parquet"
)

base.to_parquet(path, index=False)

print("\nSaved:", path)
print("\nGOAT POSTSEASON SCORING BASE V1 COMPLETE")
