import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# ===== Load RB data =====
rb = pd.read_csv(
    r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_rb_vet_data.csv"
)

# --- Column helpers (be robust to naming) ---
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

ppr_col = pick_col(rb, ["Fantasy_PPR", "fantasy_points_ppr_x", "Fantasy_PPR_x"])
games_col = pick_col(rb, ["Games_G", "games", "games_x", "G"])
exp_pg_col = pick_col(rb, ["expected_ppr_pg_curr_hist", "xpected_ppr_pg_curr_hist"])

if ppr_col is None or games_col is None or exp_pg_col is None:
    missing = []
    if ppr_col is None:   missing.append("Fantasy_PPR / fantasy_points_ppr_x")
    if games_col is None: missing.append("Games_G / games_x / G")
    if exp_pg_col is None:missing.append("expected_ppr_pg_curr_hist")
    raise ValueError(f"Missing required columns: {missing}")

# --- Target construction: per-game delta vs expected ---
rb["ppg_Fantasy_PPR"] = pd.to_numeric(rb[ppr_col], errors="coerce") / pd.to_numeric(rb[games_col], errors="coerce")
rb["delta_ppg"] = rb["ppg_Fantasy_PPR"] - pd.to_numeric(rb[exp_pg_col], errors="coerce")
rb["merge_year"] = pd.to_numeric(rb.get("merge_year"), errors="coerce")

rb = rb.dropna(subset=["delta_ppg", "merge_year"]).copy()

# ===== Fixed thresholds (±2.0 PPG) =====
LOW_THR  = -2.0
HIGH_THR =  2.0

def label_fixed(x):
    if x <= LOW_THR:  return "under"
    if x >= HIGH_THR: return "over"
    return "neutral"

rb["perf_class_str"] = rb["delta_ppg"].apply(label_fixed)

# ===== Encode classes for XGBoost =====
class_to_id = {"under": 0, "neutral": 1, "over": 2}
id_to_class = {v: k for k, v in class_to_id.items()}
rb["perf_class"] = rb["perf_class_str"].map(class_to_id)

# ===== Features =====
feature_candidates = [
    "att","yds_x","td_x","x1d_x","ybc_x","ybc_att","yac_x","yac_att","brk_tkl_x",
    "att_br","efficiency","percent_attempts_gte_eight_defenders",
    "expected_rush_yards","rush_yards_over_expected","rush_yards_over_expected_per_att",
    "rush_pct_over_expected","rushing_fumbles","rushing_fumbles_lost","rushing_epa",
    "rushing_2pt_conversions","fantasy_points_x","fantasy_points_ppr_x","games_x",
    "height_x","weight_x","draft_round_x","draft_pick_x","tgt","yds_y","td_y","x1d_y",
    "ybc_y","ybc_r","yac_y","yac_r","adot","brk_tkl_y","rec_br","drop","drop_percent",
    "receiving_fumbles","receiving_fumbles_lost","receiving_air_yards","receiving_epa",
    "receiving_2pt_conversions","racr","target_share","air_yards_share","wopr_x","tgt_sh",
    "ay_sh","yac_sh","wopr_y","ry_sh","rtd_sh","rfd_sh","dom","w8dom","yptmpa","ppr_sh",
    "offense_snaps","team_snaps","AVG","adp_percentile","adp_percentile_pos",
    "expected_ppr_pg_curr_hist","Games_G"
]
feature_cols = [c for c in feature_candidates if c in rb.columns and pd.api.types.is_numeric_dtype(rb[c])]

X_all = rb[feature_cols]
y_all = rb["perf_class"]

# ===== Time-based splits =====
train = rb["merge_year"].between(2016, 2020)
val   = rb["merge_year"].between(2021, 2023)
test  = rb["merge_year"].eq(2024)

X_train, y_train = X_all.loc[train], y_all.loc[train]
X_val,   y_val   = X_all.loc[val],   y_all.loc[val]
X_test,  y_test  = X_all.loc[test],  y_all.loc[test]

# ===== Impute =====
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_val_i   = imp.transform(X_val)
X_test_i  = imp.transform(X_test)

# ===== Optional: class weights to fight imbalance (computed from TRAIN) =====
class_counts = y_train.value_counts().reindex([0,1,2]).fillna(0)
tot = class_counts.sum()
weights = (tot / (3 * class_counts.replace(0, np.nan))).fillna(1.0).to_dict()  # inverse freq
sample_w = y_train.map(weights).values

# ===== XGBoost multiclass =====
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=3,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_i, y_train, sample_weight=sample_w)

# ===== Evaluation =====
def decode(y_int): return pd.Series(y_int).map(id_to_class)

y_val_pred  = xgb.predict(X_val_i)
y_test_pred = xgb.predict(X_test_i)

print("\nValidation Performance:\n",
      classification_report(decode(y_val), decode(y_val_pred), labels=["under","neutral","over"]))
print("Confusion Matrix (VAL):\n",
      confusion_matrix(decode(y_val), decode(y_val_pred), labels=["under","neutral","over"]))

print("\nTest Performance:\n",
      classification_report(decode(y_test), decode(y_test_pred), labels=["under","neutral","over"]))
print("Confusion Matrix (TEST):\n",
      confusion_matrix(decode(y_test), decode(y_test_pred), labels=["under","neutral","over"]))

# ===== Probability tables (Test 2024) =====
proba = xgb.predict_proba(X_test_i)  # columns in order of class IDs [0,1,2]
proba_df = pd.DataFrame({
    "prob_under":  proba[:, 0],
    "prob_neutral":proba[:, 1],
    "prob_over":   proba[:, 2],
}, index=X_test.index)

id_cols = [c for c in ["Player_fixed","display_name","FantPos","merge_year"] if c in rb.columns]
out = pd.concat([rb.loc[test, id_cols], proba_df], axis=1)

print("\nTop 15 RBs by probability of OVER-performing (Test 2024):")
print(out.sort_values("prob_over", ascending=False).head(15))

print("\nTop 15 RBs by probability of UNDER-performing (Test 2024):")
print(out.sort_values("prob_under", ascending=False).head(15))
