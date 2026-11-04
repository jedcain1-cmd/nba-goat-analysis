from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("results")

df = pd.read_parquet(
    OUT / "goat_regular_season_value_v1.parquet"
)

KEY = "career_player_id"
VALUE = "regular_value_R_minus_1_0_v1"
IMPACT = "regular_total_impact_v1"

print("=== GOAT PEAK DOMINANCE V2 ===")

assert not df.duplicated(
    [KEY, "season"]
).any()

df["season_year"] = (
    df["season"].str[:4].astype(int)
)

df["clean_player"] = (
    df["player"].astype(str)
    .str.rstrip("*").str.strip()
)

df["minutes"] = pd.to_numeric(
    df["minutes"], errors="coerce"
)

df[VALUE] = pd.to_numeric(
    df[VALUE], errors="coerce"
)

df[IMPACT] = pd.to_numeric(
    df[IMPACT], errors="coerce"
)

# Restrict impact-only peak comparisons to
# substantial playing-time seasons.
df["impact_qualified"] = (
    df["minutes"].ge(1500)
    & df[IMPACT].notna()
)

records = []

for player_id, group in df.groupby(KEY):

    group = group.sort_values(
        "season_year"
    ).copy()

    record = {
        KEY: player_id,
        "player": group["clean_player"].mode().iloc[0],
        "career_seasons": group["season"].nunique()
    }

    qualified = group[
        group["impact_qualified"]
    ]

    record["best1_impact"] = (
        qualified[IMPACT].max()
        if len(qualified) else np.nan
    )

    record["best1_value"] = (
        group[VALUE].max()
    )

    for n in [3, 5, 7]:

        # Best nonconsecutive seasons.
        values = group[VALUE].dropna()

        record[f"best{n}_value"] = (
            values.nlargest(n).sum()
            if len(values) >= n else np.nan
        )

        # Best consecutive seasons.
        best = np.nan
        best_start = np.nan

        for i in range(len(group) - n + 1):

            window = group.iloc[i:i+n]

            years = window[
                "season_year"
            ].to_numpy()

            if not np.all(np.diff(years) == 1):
                continue

            vals = window[VALUE]

            if vals.isna().any():
                continue

            total = vals.sum()

            if np.isnan(best) or total > best:
                best = total
                best_start = years[0]

        record[f"best{n}_consecutive_value"] = best
        record[f"best{n}_window_start"] = best_start

    records.append(record)

result = pd.DataFrame(records)

print("\n=== TOP 15 SINGLE-SEASON IMPACT ===")

print(
    result.nlargest(15, "best1_impact")[
        ["player", "best1_impact"]
    ].round(3).to_string(index=False)
)

for metric in [
    "best1_value",
    "best3_consecutive_value",
    "best5_consecutive_value",
    "best7_consecutive_value"
]:
    print("\n=== TOP 15:", metric, "===")

    print(
        result.nlargest(15, metric)[
            ["player", metric]
        ].round(2).to_string(index=False)
    )

print("\n=== SELECTED GOAT CANDIDATES ===")

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
    "Nikola Jokić",
    "Shaquille O'Neal",
    "Hakeem Olajuwon",
    "Kevin Garnett",
    "Dirk Nowitzki"
]

print(
    result[
        result["player"].isin(selected)
    ]
    .round(2)
    .to_string(index=False)
)

path = OUT / "goat_peak_dominance_v2.csv"

result.to_csv(path, index=False)

print("\nSaved:", path)
print("\nGOAT PEAK DOMINANCE V2 COMPLETE")
