import pandas as pd
import re
from pathlib import Path

# -------------------------------
# Paths (update if needed)
# -------------------------------
ADP_MASTER_PATH = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\cleaned_adp\FantasyPros_ADP_cleaned_2015_2024_master.csv")
RANKS_MASTER_PATH = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\cleaned_fantasy_ranks\fantasy_ranks_master_2015_2024.csv")
OUTPUT_MERGED_PATH = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\merged_dataset\merged_ranks_adp_2015_2024.csv")

# -------------------------------
# Helpers
# -------------------------------
def clean_name(x: str) -> str:
    if pd.isna(x): return ""
    s = str(x)
    s = re.sub(r'[\*\+]', ' ', s)          # remove * and +
    s = re.sub(r'\([^)]*\)', ' ', s)       # remove parentheses
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace(".", "").replace("'", "").lower()
    return s

# -------------------------------
# Load
# -------------------------------
adp = pd.read_csv(ADP_MASTER_PATH)
ranks = pd.read_csv(RANKS_MASTER_PATH)

# -------------------------------
# Normalize columns / keys
# -------------------------------
# ADP
adp_player_col = "Player" if "Player" in adp.columns else adp.columns[0]
if "year" not in adp.columns and "Year" in adp.columns:
    adp = adp.rename(columns={"Year": "year"})
if "AVG" not in adp.columns and "Avg" in adp.columns:
    adp = adp.rename(columns={"Avg": "AVG"})

adp["player_clean"] = adp[adp_player_col].astype(str).apply(clean_name)
adp["year"] = pd.to_numeric(adp["year"], errors="coerce")
adp["AVG"]  = pd.to_numeric(adp["AVG"],  errors="coerce")

# Keep only rows with valid keys & AVG
adp_keyed = adp.dropna(subset=["player_clean", "year", "AVG"]).copy()

# >>> Ensure right dataframe is UNIQUE on (player_clean, year)
adp_agg = (adp_keyed
           .groupby(["player_clean", "year"], as_index=False, dropna=False)
           .agg(AVG=("AVG", "mean")))          # collapse duplicates by mean AVG

# Optional sanity check (prints any remaining dup keys; should be 0)
dups_right = adp_agg.duplicated(["player_clean", "year"]).sum()
if dups_right:
    print(f"[WARN] ADP still has {dups_right} duplicate keys after aggregation.")

# RANKS
ranks_player_col = "Player" if "Player" in ranks.columns else ranks.columns[0]
if "year" not in ranks.columns and "Year" in ranks.columns:
    ranks = ranks.rename(columns={"Year": "year"})

ranks["player_clean"] = ranks[ranks_player_col].astype(str).apply(clean_name)
ranks["year"] = pd.to_numeric(ranks["year"], errors="coerce")

# Ensure PPR is numeric and compute PPR_rank per year if not present
# ranks["PPR"] = pd.to_numeric(ranks.get("PPR", pd.NA), errors="coerce")
# if "PPR_rank" not in ranks.columns:
#     ranks["PPR_rank"] = ranks.groupby("year")["PPR"].rank(method="min", ascending=False)

# -------------------------------
# Merge (many-to-one: left many rows OK; right MUST be unique)
# -------------------------------
merged = ranks.merge(
    adp_agg,
    on=["player_clean", "year"],
    how="inner",            # keep only players/years that appear in both
    validate="m:1"          # enforce many-to-one (will raise if right not unique)
)

# -------------------------------
# Metric: performance vs ADP
# -------------------------------
# Lower (more negative) => outperformed draft slot (finished better than drafted)
merged["performance_relative_to_adp"] = merged["PPR_rank_year"] - merged["AVG"]

# Save
merged.to_csv(OUTPUT_MERGED_PATH, index=False)
print(f"Done. Wrote {len(merged):,} merged rows to:\n{OUTPUT_MERGED_PATH}")
