# NBA GOAT Analysis — Ranking 5,000+ NBA Players

An independent basketball analytics research project that evaluates NBA career greatness using historical statistics, category-level scoring, and a transparent weighted model. The analysis covers NBA history through the 2025–26 season.

## Video Documentary

**[Watch the full NBA GOAT Analysis documentary on YouTube](https://youtu.be/-5IG6SGo7eM)**

This documentary presents the research, statistical methodology, historical comparisons, and final rankings through data visualizations and basketball analysis.

The project combines Python-based statistical modeling with data storytelling to explore one of basketball's most debated questions: Who is the greatest NBA player of all time?

## Results — Corrected V1

| Rank | Player | Final score |
|---:|---|---:|
| 1 | Michael Jordan | 87.23 |
| 2 | LeBron James | 83.67 |
| 3 | Kareem Abdul-Jabbar | 82.54 |

**Authoritative output:** [`results/goat_official_leaderboard_v1.csv`](results/goat_official_leaderboard_v1.csv). Scores are model-dependent estimates, not objective proof of who is the greatest.

## Model design

| Category | Overall weight | Internal components |
|---|---:|---|
| Postseason excellence | 30% | Career playoff value (50%), best five-season playoff stretch (30%), playoff scoring (20%) |
| Achievements | 25% | Championship contribution (45%), MVPs (25%), Finals MVPs (15%), other honors (15%) |
| Regular-season excellence | 20% | Career value (45%), traditional production (30%), advanced performance (25%) |
| Peak dominance | 20% | Best season (25%), best three-year period (30%), best five-year period (45%) |
| Longevity | 5% | Games, minutes and seasons played |

The corrected model accounts for historical award eligibility, including the introduction of Finals MVP in 1969. The frozen specification is preserved in [`results/goat_official_specification_v1.json`](results/goat_official_specification_v1.json).

## Research approach

1. Assemble and reconcile historical player, season, postseason, and awards data.
2. Compute category-specific player scores, including historical adjustments and championship attribution.
3. Combine scores using the frozen category weights.
4. Audit outputs and compare sensitivity to modeling assumptions.
5. Present the findings in an NBA analytics documentary.

## Repository layout

- `src/`: selected original Python scripts relevant to scoring, finalization, and sensitivity checks; this is not the complete pipeline.
- `results/`: **selected** official outputs and frozen model specification.
- The frozen JSON specification, rather than early planning documents, is the source of truth for the final weights.
- `requirements.txt`: dependencies from the original project.

The complete original project also contains intermediate model iterations, notebooks, large datasets, and exploratory outputs. This public-facing package intentionally excludes those bulk files; it is a **curated research archive**, not yet a one-command reproducible pipeline. Some source scripts may reference intermediate files not included here. See the original project for those dependencies.

## Interpretation and limitations

- Era-to-era comparisons are imperfect, particularly where historical tracking statistics are missing.
- Only a subset of the more than 5,000 players evaluated had complete information for all modeled features.
- The rankings depend on category definitions, weights, and historical adjustments.
- No individual player's rank was manually altered in the frozen scoring output.

## Documentary

**YouTube:** Documentary link to be added after publication.

## Technology

Python, pandas, NumPy, statistical modeling, historical NBA data reconciliation, and sensitivity analysis. See `requirements.txt` for the original environment's package list.

## Model provenance

Official leaderboard: `results/goat_official_leaderboard_v1.csv`  
Frozen model: `results/goat_official_specification_v1.json`  
Finalization script: `src/freeze_goat_corrected_v1.py`
