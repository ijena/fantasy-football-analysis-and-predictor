import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- File Path ----------------
file_path = r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\college_football_data 2014-2024\cfbd_player_season_stats_2014_2024_master.csv"

# ---------------- Load Data ----------------
college_football_data = pd.read_csv(file_path)

# ---------------- Filter Relevant Categories ----------------
relevant_stat_categories = ["passing", "rushing", "receiving", "interceptions", "fumbles"]
college_football_data = college_football_data[college_football_data["category"].isin(relevant_stat_categories)]

# ---------------- Pivot: category + stat_type → one column ----------------
# Keys to identify a player-season
id_cols = []
for c in ["player_id", "athlete_id", "id", "cfbd_id"]:
    if c in college_football_data.columns:
        id_cols.append(c)
        break
for c in ["player", "player_name", "athlete", "name"]:
    if c in college_football_data.columns and c not in id_cols:
        id_cols.append(c)
        break
if "season" in college_football_data.columns:
    season_col = "season"
elif "year" in college_football_data.columns:
    season_col = "year"
else:
    raise ValueError("No season/year column found")

# Always keep position if available
if "position" in college_football_data.columns and "position" not in id_cols:
    id_cols.append("position")

# Pivot
wide = college_football_data.pivot_table(
    index=id_cols + [season_col],
    columns=["category", "statType"],
    values="stat",
    aggfunc="sum",
    fill_value=0
).reset_index()

# Flatten MultiIndex columns → "category_statType"
wide.columns = [
    "_".join([str(c) for c in col if c not in ("", None)]) if isinstance(col, tuple) else str(col)
    for col in wide.columns
]

# ---------------- Save Final Cleaned File ----------------
wide.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\college_football_data 2014-2024\cleaned_wide_cfbd_player_season_stats_2014_2024_master.csv", index=False)
print(f"Final cleaned file saved to: {file_path}")
