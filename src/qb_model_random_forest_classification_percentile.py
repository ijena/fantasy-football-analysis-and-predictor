import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ===== Load =====
qb = pd.read_csv(
    r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_qb_vet_data.csv"
)

# ----- Build source metric for labels (per-game over/under vs expected) -----
games = pd.to_numeric(qb.get("Games_G"), errors="coerce").replace(0, np.nan)
actual_total = pd.to_numeric(qb.get("Fantasy_PPR"), errors="coerce")
expected_pg = pd.to_numeric(qb.get("expected_ppr_pg_curr_hist"), errors="coerce")

qb["ppg_Fantasy_PPR"] = actual_total / games
qb["per_game_perf_rel_expectations"] = qb["ppg_Fantasy_PPR"] - expected_pg

# year column for split
qb["merge_year"] = pd.to_numeric(qb["merge_year"], errors="coerce")

# keep rows that can form a label and have a split year
qb = qb.dropna(subset=["per_game_perf_rel_expectations", "merge_year"]).copy()

# ===== Time-aware split masks =====
train_mask = qb["merge_year"].between(2016, 2020)
val_mask   = qb["merge_year"].between(2021, 2023)
test_mask  = qb["merge_year"].eq(2024)

# ===== Percentile thresholds from TRAIN ONLY (no leakage) =====
train_perf = qb.loc[train_mask, "per_game_perf_rel_expectations"].dropna()
LOW_PCT, HIGH_PCT = 25, 75  # tweak to 20/80 if you want fewer over/under cases

low_th  = np.percentile(train_perf, LOW_PCT)
high_th = np.percentile(train_perf, HIGH_PCT)
print(f"Label thresholds (from TRAIN): under ≤ {low_th:.2f}, neutral between, over ≥ {high_th:.2f}")

def label_from_percentile(x, lo=low_th, hi=high_th):
    if x >= hi:
        return "over"
    elif x <= lo:
        return "under"
    else:
        return "neutral"

qb["perf_class"] = qb["per_game_perf_rel_expectations"].apply(label_from_percentile)

# ===== Features =====
feature_candidates = [
    "rank","qbr_total","pts_added","qb_plays","epa_total","pass","run","exp_sack","penalty","qbr_raw","sack",
    "pass_attempts","throwaways","spikes","drops","drop_pct","bad_throws","bad_throw_pct","pocket_time",
    "times_blitzed","times_hurried","times_hit","times_pressured","pressure_pct","batted_balls","on_tgt_throws",
    "on_tgt_pct","rpo_plays","rpo_yards","rpo_pass_att","rpo_pass_yards","rpo_rush_att","rpo_rush_yards",
    "pa_pass_att","pa_pass_yards","avg_time_to_throw","avg_completed_air_yards","avg_intended_air_yards",
    "avg_air_yards_differential","aggressiveness","max_completed_air_distance","avg_air_yards_to_sticks",
    "pass_yards","pass_touchdowns","interceptions_x","passer_rating","completion_percentage",
    "expected_completion_percentage","completion_percentage_above_expectation","avg_air_distance","max_air_distance",
    "sacks","sack_yards","sack_fumbles","sack_fumbles_lost","passing_air_yards","passing_yards_after_catch",
    "passing_epa","passing_2pt_conversions","pacr","dakota","carries","rushing_yards","rushing_tds",
    "rushing_fumbles_x","rushing_fumbles_lost_x","rushing_first_downs","rushing_epa_x","rushing_2pt_conversions_x",
    "fantasy_points_x","games_x","ppr_sh","height_x","weight_x","draft_round_x","draft_pick_x","age","gs",
    "ybc","ybc_att","yac","yac_att","brk_tkl","att_br","offense_snaps","team_snaps",
    "AVG","adp_percentile","adp_percentile_pos","expected_ppr_pg_curr_hist","Games_G"
]
# keep only existing numeric features
feature_cols = [c for c in feature_candidates if c in qb.columns and pd.api.types.is_numeric_dtype(qb[c])]
X_all = qb[feature_cols].copy()
y_all = qb["perf_class"].copy()

# ===== Split sets =====
X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
X_val,   y_val   = X_all.loc[val_mask],   y_all.loc[val_mask]
X_test,  y_test  = X_all.loc[test_mask],  y_all.loc[test_mask]

# Print class balances
print("\nClass counts (train):")
print(y_train.value_counts())
print("\nClass counts (val):")
print(y_val.value_counts())
print("\nClass counts (test):")
print(y_test.value_counts())

# ===== Impute features (median) =====
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_val_i   = imp.transform(X_val)
X_test_i  = imp.transform(X_test)

# ===== Model =====
clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"  # helps recall for minority classes
)
clf.fit(X_train_i, y_train)

# ===== Evaluate =====
print("\nValidation Performance:")
print(classification_report(y_val, clf.predict(X_val_i), digits=3))
print("\nTest Performance:")
print(classification_report(y_test, clf.predict(X_test_i), digits=3))

print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, clf.predict(X_test_i), labels=["under","neutral","over"]))
