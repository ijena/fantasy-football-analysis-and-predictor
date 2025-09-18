import pandas as pd
import numpy as np
import re
from pathlib import Path

# =====================
# Helper: Clean Features
# =====================
def clean_features(df, corr_threshold=0.99):
    """
    Cleans a fantasy football dataset by:
      1. Dropping exact duplicate columns
      2. Dropping highly correlated columns
    """
    dropped_columns = {"duplicates": [], "correlated": []}

    # Drop exact duplicate columns
    duplicates = df.T.duplicated()
    df_cleaned = df.loc[:, ~duplicates]

    # Drop highly correlated
    corr_matrix = df_cleaned.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_cols = [col for col in upper.columns if any(upper[col] > corr_threshold) and col not in ["merge_year"]]
    df_cleaned = df_cleaned.drop(columns=corr_cols, errors="ignore")
    
    return df_cleaned

# =====================
# Paths
# =====================
BASE = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data")

# Load datasets
master_passing_df   = pd.read_csv(BASE / "nflverse_data" / "master_passing_data.csv").drop(columns=["Unnamed: 0"], errors="ignore")
master_receiving_df = pd.read_csv(BASE / "nflverse_data" / "master_receiving_data.csv")
master_rushing_df   = pd.read_csv(BASE / "nflverse_data" / "master_rushing_data.csv")

college_qbr_df      = pd.read_csv(BASE / "nflverse_data" / "college_qbr_data.csv")
college_stats_df    = pd.read_csv(BASE / "college_football_data 2014-2024" / "cleaned_wide_cfbd_player_season_stats_2014_2024_master.csv")

adp_master_df       = pd.read_csv(BASE / "cleaned_adp" / "FantasyPros_ADP_cleaned_2015_2024_master.csv")
snap_counts_df      = pd.read_csv(BASE / "nflverse_data" / "season_snap_count_data.csv")
combine_data_df     = pd.read_csv(BASE / "nflverse_data" / "combine_data.csv")

merged_fantasy_rank_adp_with_expected_points_df = pd.read_csv(
    BASE / "merged_dataset" / "merged_with_expected_pg_and_season.csv"
)

# Clean NFL dataframes
cleaned_master_passing_df   = clean_features(master_passing_df)
cleaned_master_receiving_df = clean_features(master_receiving_df)
cleaned_master_rushing_df   = clean_features(master_rushing_df)

# Merge college stats + QBR
master_college_stats = pd.merge(
    college_stats_df,
    college_qbr_df,
    left_on=["season", "player"],
    right_on=["season", "name_short"],
    how="left"
)
cleaned_master_college_stats = clean_features(master_college_stats)

# =====================
# Feature Columns (NFL expected/adp side)
# =====================
feature_cols = [
    "AVG", "adp_rank", "adp_percentile", "adp_percentile_pos",
    "POS_group", "FantPos", "year","Player_fixed",
    "expected_ppr_pg_prev", "expected_ppr_season_prev",
    "expected_ppr_pg_curr_hist", "expected_ppr_season_curr_hist",
    "Games_G"
]
merged_expected_points_adp_df = merged_fantasy_rank_adp_with_expected_points_df[feature_cols]

# =====================
# Build QB Veteran Dataset
# =====================
qb_passing_rushing_df = cleaned_master_passing_df.merge(
    cleaned_master_rushing_df[cleaned_master_rushing_df["position"]=="QB"],
    how="left",
    on =["display_name","season"]
)
qb_snap_count_passing_rushing_df = qb_passing_rushing_df.merge(
    snap_counts_df[snap_counts_df["position"]=="QB"],
    how="left",
    left_on=["display_name","season"],
    right_on=["player","season"]
)

#ensuring that adp from season N gets merged with stats from season N-1 to avoid leakage
qb_snap_count_passing_rushing_df["merge_year"] = qb_snap_count_passing_rushing_df["season"] + 1
# qb_snap_count_passing_rushing_df.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\test.csv")

qb_expected_points_adp_snap_count_passing_rushing_df = qb_snap_count_passing_rushing_df.merge(
    merged_expected_points_adp_df[merged_expected_points_adp_df["POS_group"]=="QB"],
    how="right",
    left_on=["display_name","merge_year"],
    right_on=["Player_fixed","year"]
)

# Veteran dataset

qb_expected_points_adp_snap_count_passing_rushing_df["rookie_x"] = qb_expected_points_adp_snap_count_passing_rushing_df["rookie_x"].fillna(0)
master_qb_vet_model_df = qb_expected_points_adp_snap_count_passing_rushing_df[
    qb_expected_points_adp_snap_count_passing_rushing_df["rookie_x"] == 0
]
master_qb_vet_model_df = clean_features(master_qb_vet_model_df)
master_qb_vet_model_df.to_csv(BASE / "model_data" / "master_qb_vet_data.csv", index=False)

#build veteran RB dataset

rb_rushing_receiving_df = cleaned_master_rushing_df[cleaned_master_rushing_df["position"]=="RB"].merge(
    cleaned_master_receiving_df[cleaned_master_receiving_df["position"]=="RB"],how="left",on=["display_name","season"])

# rb_rushing_receiving_df.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\test.csv")

rb_snap_count_rushing_receiving_df = rb_rushing_receiving_df.merge(
    snap_counts_df[snap_counts_df["position"]=="RB"],
    how="left",
    left_on=["display_name","season"],
    right_on=["player","season"]
)
# rb_snap_count_rushing_receiving_df.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\test.csv")

#ensuring that ADP from season N gets merged with stats from season N-1
rb_snap_count_rushing_receiving_df["merge_year"] = rb_snap_count_rushing_receiving_df["season"] + 1


rb_expected_points_adp_snap_count_rushing_receiving_df = rb_snap_count_rushing_receiving_df.merge(
    merged_expected_points_adp_df[merged_expected_points_adp_df["POS_group"]=="RB"],
    how="right",
    left_on=["display_name","merge_year"],
    right_on=["Player_fixed","year"]
)

# Veteran dataset

rb_expected_points_adp_snap_count_rushing_receiving_df["rookie_x"] = rb_expected_points_adp_snap_count_rushing_receiving_df["rookie_x"].fillna(0)
master_rb_vet_model_df = rb_expected_points_adp_snap_count_rushing_receiving_df[
    rb_expected_points_adp_snap_count_rushing_receiving_df["rookie_x"] == 0
]
master_rb_vet_model_df = clean_features(master_rb_vet_model_df)
master_rb_vet_model_df.to_csv(BASE / "model_data" / "master_rb_vet_data.csv", index=False)

#creating TE veteran dataset
te_receiving_df = cleaned_master_receiving_df[cleaned_master_receiving_df["position"]=="TE"]

# rb_rushing_receiving_df.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\test.csv")

te_snap_count_receiving_df = te_receiving_df.merge(
    snap_counts_df[snap_counts_df["position"]=="TE"],
    how="left",
    left_on=["display_name","season"],
    right_on=["player","season"]
)
# rb_snap_count_rushing_receiving_df.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\test.csv")

#ensuring that ADP from season N gets merged with stats from season N-1
te_snap_count_receiving_df["merge_year"] = te_snap_count_receiving_df["season"] + 1


te_expected_points_adp_snap_count_receiving_df = te_snap_count_receiving_df.merge(
    merged_expected_points_adp_df[merged_expected_points_adp_df["POS_group"]=="TE"],
    how="right",
    left_on=["display_name","merge_year"],
    right_on=["Player_fixed","year"]
)

# Veteran dataset

te_expected_points_adp_snap_count_receiving_df["rookie"] = te_expected_points_adp_snap_count_receiving_df["rookie"].fillna(0)
master_te_vet_model_df = te_expected_points_adp_snap_count_receiving_df[
te_expected_points_adp_snap_count_receiving_df["rookie"] == 0
]
master_te_vet_model_df = clean_features(master_te_vet_model_df)
master_te_vet_model_df.to_csv(BASE / "model_data" / "master_te_vet_data.csv", index=False)

# =====================
# Rookie QB: Merge Combine + 4yrs College
# =====================
def _norm_name(s):
    if pd.isna(s): return s
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)      # drop punctuation
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _ensure_player_clean(df: pd.DataFrame, candidates):
    df = df.copy()
    for c in candidates:
        if c in df.columns:
            if c != "player_clean":
                df["player_clean"] = df[c]
            break
    if "player_clean" not in df.columns:
        raise ValueError(f"No player name column found; expected {candidates}")
    df["player_clean"] = df["player_clean"].astype(str).map(_norm_name)
    return df

# Rookie QB NFL index
rook_qb_nfl = qb_expected_points_adp_snap_count_passing_rushing_df.copy()
rook_qb_nfl = _ensure_player_clean(rook_qb_nfl, ["Player_fixed","name_display","player","player_clean"])
rook_qb_nfl = rook_qb_nfl.rename(columns={"season":"nfl_season"})
rook_qb_nfl["rookie_year"] = rook_qb_nfl["nfl_season"]
rook_qb_nfl = rook_qb_nfl[rook_qb_nfl.get("rookie_x",1) == 1].copy()

# College prep
college_raw = _ensure_player_clean(college_stats_df, ["player","name_short","name_display","player_clean"])
qbr_raw     = _ensure_player_clean(college_qbr_df, ["name_short","player","name_display","player_clean"])

qbr_cols_keep = ["player_clean","season"]
for cand in ["QBR","qbr","espn_qbr"]:
    if cand in qbr_raw.columns:
        qbr_cols_keep.append(cand); break

college_with_qbr = college_raw.merge(
    qbr_raw[qbr_cols_keep].drop_duplicates(),
    on=["player_clean","season"], how="left"
)

# filter college for QBs if possible
for pos_col in ["position","pos","POS","FantPos","POS_group"]:
    if pos_col in college_with_qbr.columns:
        college_with_qbr = college_with_qbr[college_with_qbr[pos_col].astype(str).str.upper().str.contains("QB")]
        break

# merge rookies ↔ college
rook_key = rook_qb_nfl[["player_clean","rookie_year"]].drop_duplicates()
college_join = college_with_qbr.merge(rook_key, on="player_clean", how="inner")
college_join = college_join[college_join["season"] <= college_join["rookie_year"] - 1].copy()

# Select numeric columns
id_like = {"player_clean","season","rookie_year"}
num_cols = [c for c in college_join.columns if c not in id_like and pd.api.types.is_numeric_dtype(college_join[c])]

# recency rank (cyr1 = most recent college season)
wide_base = college_join[["player_clean","season","rookie_year"] + num_cols].copy()
wide_base["__rank_recent"] = wide_base.groupby("player_clean")["season"].rank(method="first", ascending=False).astype(int)
wide_base = wide_base[wide_base["__rank_recent"] <= 4].copy()
wide_base["cyr"] = "cyr" + wide_base["__rank_recent"].astype(str)

tidy = wide_base.melt(
    id_vars=["player_clean","rookie_year","cyr"],
    value_vars=num_cols,
    var_name="stat",
    value_name="val"
)
tidy["stat_cyr"] = tidy["stat"] + "_" + tidy["cyr"]
rook_college_wide = tidy.pivot_table(
    index=["player_clean","rookie_year"],
    columns="stat_cyr", values="val", aggfunc="last"
).reset_index()

# merge onto rookie NFL table
rook_qb_with_college4 = rook_qb_nfl.merge(rook_college_wide, on=["player_clean","rookie_year"], how="left")

# Save rookie QB dataset
out_rook_qb = BASE / "model_data" / "rookies_QB_with_college4yr_wide.csv"
rook_qb_with_college4.to_csv(out_rook_qb, index=False)
print("✅ Saved rookie QB with 4yr college features →", out_rook_qb)
