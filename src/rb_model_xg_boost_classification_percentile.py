import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# ===== Load RB Data =====
rb = pd.read_csv(
    r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_rb_vet_data.csv"
)

# --- Build target: per-game over/under vs expected ---
rb["ppg_Fantasy_PPR"] = pd.to_numeric(rb.get("Fantasy_PPR"), errors="coerce") / pd.to_numeric(rb.get("Games_G"), errors="coerce")
rb["per_game_perf_rel_expectations"] = rb["ppg_Fantasy_PPR"] - pd.to_numeric(rb.get("expected_ppr_pg_curr_hist"), errors="coerce")
rb["merge_year"] = pd.to_numeric(rb.get("merge_year"), errors="coerce")

rb = rb.dropna(subset=["per_game_perf_rel_expectations", "merge_year"]).copy()

# ===== Percentile thresholds from TRAIN only (30/70) =====
train_mask = rb["merge_year"].between(2016, 2020)
train_perf = rb.loc[train_mask, "per_game_perf_rel_expectations"].dropna()
low_th  = np.percentile(train_perf, 30)
high_th = np.percentile(train_perf, 70)
print(f"Label thresholds from TRAIN: under ≤ {low_th:.2f}, neutral between, over ≥ {high_th:.2f}")

def label_from_perf(x):
    if x <= low_th:  return "under"
    if x >= high_th: return "over"
    return "neutral"

rb["perf_class_str"] = rb["per_game_perf_rel_expectations"].apply(label_from_perf)

# ===== Fixed class encoding for XGB =====
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

# ===== Time splits =====
train = rb["merge_year"].between(2016, 2020)
val   = rb["merge_year"].between(2021, 2023)
test  = rb["merge_year"].eq(2024)

X_train, y_train = X_all.loc[train], y_all.loc[train]
X_val,   y_val   = X_all.loc[val],   y_all.loc[val]
X_test,  y_test  = X_all.loc[test],  y_all.loc[test]

# ===== Imputation =====
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_val_i   = imp.transform(X_val)
X_test_i  = imp.transform(X_test)

# ===== XGBoost (multiclass) =====
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_i, y_train)

# ===== Evaluation (use human-readable labels) =====
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
proba = xgb.predict_proba(X_test_i)  # shape (N, 3) aligned with class_to_id order 0:under,1:neutral,2:over
proba_df = pd.DataFrame({
    "prob_under":  proba[:, class_to_id["under"]],
    "prob_neutral":proba[:, class_to_id["neutral"]],
    "prob_over":   proba[:, class_to_id["over"]],
}, index=X_test.index)

id_cols = [c for c in ["Player_fixed","display_name","FantPos","merge_year"] if c in rb.columns]
out = pd.concat([rb.loc[test, id_cols], proba_df], axis=1)

print("\nTop 15 RBs by probability of OVER-performing (Test 2024):")
print(out.sort_values("prob_over", ascending=False).head(15))

print("\nTop 15 RBs by probability of UNDER-performing (Test 2024):")
print(out.sort_values("prob_under", ascending=False).head(15))
