import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

# --- load ---
qb = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_qb_vet_data.csv")

# --- build per-game label source ---
qb["ppg_Fantasy_PPR"] = pd.to_numeric(qb["Fantasy_PPR"], errors="coerce") / pd.to_numeric(qb["Games_G"], errors="coerce").replace(0, np.nan)
qb["per_game_perf_rel_expectations"] = qb["ppg_Fantasy_PPR"] - pd.to_numeric(qb["expected_ppr_pg_curr_hist"], errors="coerce")

# 3-class labels via ±2.0 PPR/game threshold
def label_from_perf(x, th=3.0):
    if pd.isna(x): return np.nan
    if x >= th:    return "over"
    if x <= -th:   return "under"
    return "neutral"

qb["perf_class"] = qb["per_game_perf_rel_expectations"].apply(label_from_perf)

# keep valid rows
qb["merge_year"] = pd.to_numeric(qb["merge_year"], errors="coerce")
qb = qb.dropna(subset=["perf_class", "merge_year"]).copy()

# --- features ---
feature_candidates = [
    "att","yds_x","td_x","x1d_x","ybc_x","ybc_att","yac_x","yac_att","brk_tkl_x",
    "att_br","efficiency","percent_attempts_gte_eight_defenders",
    "expected_rush_yards","rush_yards_over_expected","rush_yards_over_expected_per_att",
    "rush_pct_over_expected","rushing_fumbles","rushing_fumbles_lost","rushing_epa",
    "rushing_2pt_conversions","fantasy_points_x","fantasy_points_ppr_x","games_x",
    "height_x","weight_x","draft_round_x","draft_pick_x","tgt","yds_y","td_y","x1d_y",
    "ybc_y","ybc_r","yac_y","yac_r","adot","brk_tkl_y","rec_br","drop","drop_percent","int","rat",
    "receiving_fumbles","receiving_fumbles_lost","receiving_air_yards","receiving_epa",
    "receiving_2pt_conversions","racr","target_share","air_yards_share","wopr_x","tgt_sh",
    "ay_sh","yac_sh","wopr_y","ry_sh","rtd_sh","rfd_sh","dom","w8dom","yptmpa","ppr_sh",
    "offense_snaps","team_snaps","AVG","adp_percentile","adp_percentile_pos",
    "expected_ppr_pg_curr_hist","Games_G"
]
feature_cols = [c for c in feature_candidates if c in qb.columns and pd.api.types.is_numeric_dtype(qb[c])]
X_all = qb[feature_cols].copy()

# --- encode y -> integers 0..2 ---
label_map = {"under": 0, "neutral": 1, "over": 2}
inv_label_map = {v: k for k, v in label_map.items()}
y_all = qb["perf_class"].map(label_map).astype(int)

# --- time-aware split ---
train_mask = qb["merge_year"].between(2016, 2020)
val_mask   = qb["merge_year"].between(2021, 2023)
test_mask  = qb["merge_year"].eq(2024)

X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
X_val,   y_val   = X_all.loc[val_mask],   y_all.loc[val_mask]
X_test,  y_test  = X_all.loc[test_mask],  y_all.loc[test_mask]

# --- impute features (median) ---
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_val_i   = imp.transform(X_val)
X_test_i  = imp.transform(X_test)

# --- XGBoost (multi-class) ---
clf = XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",  # probabilities; use argmax for class
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train_i, y_train, eval_set=[(X_val_i, y_val)], verbose=False)

# --- evaluate ---
def report(split_name, X_i, y_true):
    proba = clf.predict_proba(X_i)
    y_pred_int = proba.argmax(axis=1)
    y_pred_lbl = pd.Series(y_pred_int).map(inv_label_map)

    print(f"\n{split_name} classification report:")
    print(classification_report(
        [inv_label_map[i] for i in y_true],
        list(y_pred_lbl),
        labels=["under","neutral","over"],
        digits=3
    ))
    print("Confusion matrix:\n", confusion_matrix(
        [inv_label_map[i] for i in y_true],
        list(y_pred_lbl),
        labels=["under","neutral","over"]
    ))

report("Validation", X_val_i, y_val.values)
report("Test",       X_test_i, y_test.values)

joblib.dump(clf,r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\qb_model_xg_boost_classification_percentile.pkl")