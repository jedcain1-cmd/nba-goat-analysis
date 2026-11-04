from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("results")
KEY = "career_player_id"

print("=== GOAT ACCOLADES PROFILE V1 ===")

rankings = pd.read_csv(
    OUT / "goat_provisional_rankings_v2.csv"
)

honors = pd.read_parquet(
    OUT / "career_honors_v1.parquet"
)

mvp = pd.read_csv(
    OUT / "career_accomplishments_v2_mvp_changes.csv"
)

assert not rankings[KEY].duplicated().any()
assert not honors[KEY].duplicated().any()
assert not mvp[KEY].duplicated().any()

honor_cols = [
    "all_nba_first",
    "all_nba_second",
    "all_nba_third",
    "all_def_first",
    "all_def_second",
    "finals_mvp",
    "all_nba_total",
    "all_def_total"
]

mvp_cols = [
    "mvp_wins_v2",
    "mvp_voting_share_v2"
]

required = [KEY] + honor_cols

assert all(
    col in honors.columns
    for col in required
)

assert all(
    col in mvp.columns
    for col in [KEY] + mvp_cols
)

profile = rankings[
    [
        KEY,
        "clean_player",
        "combined_rank",
        "combined_value"
    ]
].merge(
    honors[[KEY] + honor_cols],
    on=KEY,
    how="left",
    validate="one_to_one",
    indicator="_honors_match"
).merge(
    mvp[[KEY] + mvp_cols],
    on=KEY,
    how="left",
    validate="one_to_one",
    indicator="_mvp_match"
)

print("\n=== IDENTITY COVERAGE ===")

print(
    "Honors matches:",
    profile["_honors_match"].eq("both").sum()
)

print(
    "MVP matches:",
    profile["_mvp_match"].eq("both").sum()
)

print("\n=== ACCOLADE COVERAGE ===")

for col in honor_cols + mvp_cols:
    print(
        col,
        "| nonmissing:",
        profile[col].notna().sum(),
        "| total:",
        profile[col].sum()
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

cols = [
    "clean_player",
    "combined_rank",
    "mvp_wins_v2",
    "mvp_voting_share_v2",
    "finals_mvp",
    "all_nba_first",
    "all_nba_second",
    "all_nba_third",
    "all_def_first",
    "all_def_second"
]

print(
    profile[
        profile["clean_player"].isin(selected)
    ][cols]
    .sort_values("combined_rank")
    .round(2)
    .to_string(index=False)
)

path = OUT / "goat_accolades_profile_v1.csv"

profile.to_csv(path, index=False)

print("\nSaved:", path)
print("\nGOAT ACCOLADES PROFILE V1 COMPLETE")
