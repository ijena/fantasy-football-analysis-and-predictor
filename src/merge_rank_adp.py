import pandas as pd
from pathlib import Path

# ====== Paths ======
BASE = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data")
ADP_PATH   = BASE / "cleaned_adp" / "FantasyPros_ADP_cleaned_2015_2024_master.csv"
RANKS_PATH = BASE / "cleaned_fantasy_ranks" / "fantasy_ranks_master_2015_2024.csv"
OUT_PATH   = BASE / "merged_dataset"/ "merged_adp_ranks_2015_2024_with_metrics.csv"

# ====== Load ======
adp   = pd.read_csv(ADP_PATH)
ranks = pd.read_csv(RANKS_PATH)

print("ADP:", adp.shape, "Ranks:", ranks.shape)

# ---------- helpers ----------
def ensure_cols(df: pd.DataFrame, need: list[str], name: str):
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

def pick_pos_col(df: pd.DataFrame, preferred=None):
    for c in preferred or []:
        if c in df.columns:
            return c
    for c in ["POS_group", "FantPos", "POS", "Pos"]:
        if c in df.columns:
            return c
    return None

# ====== Minimal column checks ======
# ADP created by your cleaning script
ensure_cols(
    adp,
    ["year","player_clean","Player_fixed","AVG","adp_rank","adp_percentile"],
    "ADP"
)
# Fantasy ranks created by your ranks cleaning script (names can vary a bit)
# We'll detect PPR and POS columns with fallbacks later, but we need at least the keys:
ensure_cols(
    ranks,
    ["year","player_clean","Player_fixed"],
    "Ranks"
)

# ====== Deduplicate ADP to 1 row per (year, player_clean) ======
# Average the numeric ADP fields if duplicates exist; keep first for others.
adp_numeric_candidates = [c for c in ["AVG","adp_rank","adp_percentile","adp_percentile_pos"] if c in adp.columns]
adp_agg_spec = {c: "mean" for c in adp_numeric_candidates}
adp_agg_spec.update({
    "Player_fixed": "first",
    # keep a POS column from ADP if present
    **({"POS_group": "first"} if "POS_group" in adp.columns else {})
})

adp_agg = (
    adp.groupby(["year","player_clean"], as_index=False)
       .agg(adp_agg_spec)
)

# If adp_rank got averaged to float, tidy it
if "adp_rank" in adp_agg.columns:
    adp_agg["adp_rank"] = pd.to_numeric(adp_agg["adp_rank"], errors="coerce").round().astype("Int64")

# ====== Deduplicate RANKS to 1 row per (year, player_clean) but KEEP ALL STATS ======
# Strategy: choose ONE row per key (prefer highest Fantasy_PPR; if absent, lowest PPR_rank_year; else first),
# then take that entire row so all stats are retained.
def choose_idx(group: pd.DataFrame) -> int:
    # Try Fantasy_PPR (higher is better)
    ppr_col = None
    for c in ["Fantasy_PPR","PPR","PPR_total","PPR Points","Fantasy PPR","PPR_"]:
        if c in group.columns:
            ppr_col = c
            break
    if ppr_col is not None:
        return group[ppr_col].astype("float64").idxmax()

    # Fallback: PPR rank (lower is better)
    if "PPR_rank_year" in group.columns:
        return group["PPR_rank_year"].astype("float64").idxmin()

    # Last resort: first row
    return group.index[0]

if ranks.duplicated(["year","player_clean"]).any():
    idx = (
        ranks.groupby(["year","player_clean"], group_keys=False)
             .apply(lambda g: pd.Series({"_pick": choose_idx(g)}))["_pick"]
             .astype(int)
             .tolist()
    )
    ranks_dedup = ranks.loc[idx].copy()
else:
    ranks_dedup = ranks.copy()

# ====== Merge ======
merged = adp_agg.merge(
    ranks_dedup[["year","player_clean","Fantasy_FantPt","Fantasy_PPR","Fantasy_DKPt","Fantasy_FDPt","Fantasy_VBD",
                 "Fantasy_PosRank","Fantasy_OvRank","PPR_rank_year","ppr_percentile_overall",
                 "ppr_percentile_pos"]],
    on=["year","player_clean"],
    how="inner",
)

# Determine POS column to use for position-based metrics
pos_col = None
for c in ["POS_group_rank","POS_group_adp","POS_group","FantPos","POS","Pos"]:
    if c in merged.columns:
        pos_col = c
        break

# Determine PPR column name (from ranks)
ppr_col = None
for c in ["Fantasy_PPR","PPR","PPR_total","PPR Points","Fantasy PPR","PPR_"]:
    if c in merged.columns:
        ppr_col = c
        break

# Sanity: if needed columns are missing for metrics, create placeholders to avoid KeyErrors
if "PPR_rank_year" not in merged.columns:
    merged["PPR_rank_year"] = pd.NA
if ppr_col is None:
    # create a placeholder numeric so downstream math works
    ppr_col = "_PPR_placeholder"
    merged[ppr_col] = pd.NA

# ====== Metrics ======
# 1) Rank delta: positive => overperformed (finished better than drafted)
merged["perf_rank_delta"] = pd.to_numeric(merged["adp_rank"], errors="coerce") - pd.to_numeric(merged["PPR_rank_year"], errors="coerce")

# 2) Percentile delta (overall)
if "ppr_percentile_overall" in merged.columns:
    merged["perf_percentile_overall"] = pd.to_numeric(merged["ppr_percentile_overall"], errors="coerce") - pd.to_numeric(merged["adp_percentile"], errors="coerce")
else:
    merged["perf_percentile_overall"] = pd.NA

# 3) Percentile delta (within position) if ADP position percentile exists
if "adp_percentile_pos" in merged.columns and "ppr_percentile_pos" in merged.columns:
    merged["perf_percentile_pos"] = pd.to_numeric(merged["ppr_percentile_pos"], errors="coerce") - pd.to_numeric(merged["adp_percentile_pos"], errors="coerce")
else:
    merged["perf_percentile_pos"] = pd.NA

# 4) Points vs position/year average & vs year average (use ranks PPR)
if pos_col is not None and ppr_col is not None:
    merged["ppr_vs_pos_year_avg"] = (
        pd.to_numeric(merged[ppr_col], errors="coerce")
        - merged.groupby(["year", pos_col])[ppr_col].transform(lambda s: pd.to_numeric(s, errors="coerce").mean())
    )
else:
    merged["ppr_vs_pos_year_avg"] = pd.NA

if ppr_col is not None:
    merged["ppr_vs_year_avg"] = (
        pd.to_numeric(merged[ppr_col], errors="coerce")
        - merged.groupby("year")[ppr_col].transform(lambda s: pd.to_numeric(s, errors="coerce").mean())
    )
else:
    merged["ppr_vs_year_avg"] = pd.NA

# ====== Quick preview ======
preview_cols = [
    "year",
    "Player_fixed_adp", "Player_fixed_rank",
    "AVG", "adp_rank", "adp_percentile",
    ppr_col, "PPR_rank_year",
    "perf_rank_delta", "perf_percentile_overall", "perf_percentile_pos",
    "ppr_vs_pos_year_avg", "ppr_vs_year_avg"
]
have_cols = [c for c in preview_cols if c in merged.columns]
print("\nMerged shape:", merged.shape)
print(merged.sort_values("perf_percentile_overall", ascending=False)[have_cols].head(10))

# ====== Save ======
merged.to_csv(OUT_PATH, index=False)
print(f"\n✅ Saved merged + metrics → {OUT_PATH}")
