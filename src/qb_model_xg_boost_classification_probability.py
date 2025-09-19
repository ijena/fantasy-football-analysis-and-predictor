import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

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

qb["merge_year"] = pd.to_numeric(qb["merge_year"], errors="coerce")
qb = qb.dropna(subset=["per_game_perf_rel_expectations", "merge_year"]).copy()

# ===== Time-aware split =====
train_mask = qb["merge_year"].between(2016, 2020)
val_mask   = qb["merge_year"].between(2021, 2023)
test_mask  = qb["merge_year"].eq(2024)

# ===== Percentile labels from TRAIN ONLY (20/80) =====
train_perf = qb.loc[train_mask, "per_game_perf_rel_expectations"].dropna()
LOW_PCT, HIGH_PCT = 20, 80
low_th  = np.percentile(train_perf, LOW_PCT)
high_th = np.percentile(train_perf, HIGH_PCT)
print(f"Label thresholds (from TRAIN): under ≤ {low_th:.2f}, neutral between, over ≥ {high_th:.2f}")

def label_from_percentile(x, lo=low_th, hi=high_th):
    if x >= hi:  return "over"
    if x <= lo:  return "under"
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
feature_cols = [c for c in feature_candidates if c in qb.columns and pd.api.types.is_numeric_dtype(qb[c])]

X_all = qb[feature_cols].copy()
y_all = qb["perf_class"].copy()

# ===== Split sets =====
X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
X_val,   y_val   = X_all.loc[val_mask],   y_all.loc[val_mask]
X_test,  y_test  = X_all.loc[test_mask],  y_all.loc[test_mask]

# ===== Encode labels to ints; keep mapping for readability =====
label_map = {"under":0, "neutral":1, "over":2}
inv_label_map = {v:k for k,v in label_map.items()}
y_train_enc = y_train.map(label_map).astype(int)
y_val_enc   = y_val.map(label_map).astype(int)
y_test_enc  = y_test.map(label_map).astype(int)

# ===== Impute features (median) =====
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_val_i   = imp.transform(X_val)
X_test_i  = imp.transform(X_test)

# ===== XGBoost: output probabilities (multi:softprob) =====
clf = XGBClassifier(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.07,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",   # probabilities for each class
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train_i, y_train_enc, eval_set=[(X_val_i, y_val_enc)], verbose=False)

# ===== Evaluate (hard labels via argmax) =====
from sklearn.metrics import classification_report, confusion_matrix
def eval_split(name, X_i, y_enc):
    proba = clf.predict_proba(X_i)                  # shape (n, 3)
    pred_enc = proba.argmax(axis=1)                 # 0/1/2
    pred_lbl = pd.Series(pred_enc).map(inv_label_map)
    true_lbl = pd.Series(y_enc).map(inv_label_map)
    print(f"\n{name} Performance:")
    print(classification_report(true_lbl, pred_lbl, labels=["under","neutral","over"], digits=3))
    print("Confusion Matrix:\n", confusion_matrix(true_lbl, pred_lbl, labels=["under","neutral","over"]))
    return proba, pred_lbl

val_proba, _  = eval_split("Validation", X_val_i, y_val_enc)
test_proba, _ = eval_split("Test",        X_test_i, y_test_enc)

# ===== Build a probability table for TEST year =====
# Keep useful identifiers if present (won’t break if absent)
id_cols = [c for c in ["Player_fixed","name_display","display_name","team_abb","merge_year","year","FantPos"] if c in qb.columns]
test_ids = qb.loc[test_mask, id_cols].reset_index(drop=True)

# Wrap probabilities into a DataFrame (columns in label order)
test_prob_df = pd.DataFrame(test_proba, columns=[inv_label_map[i] for i in range(3)])
# Add convenience columns: prob_over / prob_under
test_prob_df = test_prob_df.rename(columns={"over":"prob_over","neutral":"prob_neutral","under":"prob_under"})

test_out = pd.concat([test_ids, test_prob_df], axis=1)
# Optional: rank by probability of overperformance
if "prob_over" in test_out.columns:
    test_out = test_out.sort_values("prob_over", ascending=False)

print("\nTop 15 by probability of OVER-performing (Test 2024):")
print(test_out.head(15))

# ===== (Optional) save to CSV =====
# test_out.to_csv(r"C:\path\to\save\qb_overprob_test2024.csv", index=False)
