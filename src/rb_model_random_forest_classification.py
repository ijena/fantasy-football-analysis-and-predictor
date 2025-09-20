import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ===== Load RB dataset =====
rb = pd.read_csv(
    r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_rb_vet_data.csv"
)

# ===== Build per-game performance vs expected =====
# Safe numerics
games = pd.to_numeric(rb.get("Games_G"), errors="coerce").replace(0, np.nan)
actual_total = pd.to_numeric(rb.get("Fantasy_PPR"), errors="coerce")
expected_pg = pd.to_numeric(rb.get("expected_ppr_pg_curr_hist"), errors="coerce")

rb["ppg_Fantasy_PPR"] = actual_total / games
rb["per_game_perf_rel_expectations"] = rb["ppg_Fantasy_PPR"] - expected_pg

# year split column
rb["merge_year"] = pd.to_numeric(rb.get("merge_year"), errors="coerce")

# keep rows that can form labels and have year
rb = rb.dropna(subset=["per_game_perf_rel_expectations", "merge_year"]).copy()

# ===== Labels: ±3 PPR per game around expected =====
def label_from_perf(x, th=2.0):
    if x >= th:
        return "over"
    elif x <= -th:
        return "under"
    else:
        return "neutral"

rb["perf_class"] = rb["per_game_perf_rel_expectations"].apply(label_from_perf)

# ===== Feature columns =====
# (Cleaned to avoid the stray tab in "avg_time_to_los\tavg_rush_yards")
feature_candidates = [
    "att","yds_x","td_x","x1d_x","ybc_x","ybc_att","yac_x","yac_att","brk_tkl_x",
    "att_br","efficiency","percent_attempts_gte_eight_defenders",
    "expected_rush_yards","rush_yards_over_expected","rush_yards_over_expected_per_att",
    "rush_pct_over_expected","rushing_fumbles","rushing_fumbles_lost","rushing_epa",
    "rushing_2pt_conversions","fantasy_points_x","fantasy_points_ppr_x","games_x",
    "height_x","weight_x","draft_round_x","draft_pick_x","tgt","yds_y","td_y","x1d_y",
    "ybc_y","ybc_r","yac_y","yac_r","adot","brk_tkl_y","rec_br","drop","drop_percent",
    "int","rat","receiving_fumbles","receiving_fumbles_lost","receiving_air_yards",
    "receiving_epa","receiving_2pt_conversions","racr","target_share","air_yards_share",
    "wopr_x","tgt_sh","ay_sh","yac_sh","wopr_y","ry_sh","rtd_sh","rfd_sh","dom","w8dom",
    "yptmpa","ppr_sh","offense_snaps","team_snaps","AVG","adp_percentile",
    "adp_percentile_pos","expected_ppr_pg_curr_hist","Games_G"
]
# keep only numeric features that actually exist
feature_cols = [c for c in feature_candidates if c in rb.columns and pd.api.types.is_numeric_dtype(rb[c])]

X_all = rb[feature_cols].copy()
y_all = rb["perf_class"].copy()

# ===== Time-aware splits =====
train_mask = rb["merge_year"].between(2016, 2020)
val_mask   = rb["merge_year"].between(2021, 2023)
test_mask  = rb["merge_year"].eq(2024)

X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
X_val,   y_val   = X_all.loc[val_mask],   y_all.loc[val_mask]
X_test,  y_test  = X_all.loc[test_mask],  y_all.loc[test_mask]

# ===== Imputation =====
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_val_i   = imp.transform(X_val)
X_test_i  = imp.transform(X_test)

# ===== Random Forest classifier (probabilities) =====
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"   # help minority (over/under) classes
)
rf.fit(X_train_i, y_train)

# ===== Evaluate (hard labels) =====
print("Validation Performance:\n", classification_report(y_val, rf.predict(X_val_i)))
print("Test Performance:\n", classification_report(y_test, rf.predict(X_test_i)))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, rf.predict(X_test_i), labels=["under","neutral","over"]))

# ===== Probability outputs for Test (2024) =====
proba_test = rf.predict_proba(X_test_i)              # columns follow rf.classes_ order
class_order = list(rf.classes_)                      # e.g., ['neutral','over','under'] (alphabetical)
proba_df = pd.DataFrame(proba_test, columns=[f"prob_{c}" for c in class_order])

# Helpful IDs if present
id_cols = [c for c in ["Player_fixed","name_display","display_name","team_abb","merge_year","year","FantPos"] if c in rb.columns]
test_ids = rb.loc[test_mask, id_cols].reset_index(drop=True)

out = pd.concat([test_ids, proba_df], axis=1)

# Create stable columns prob_over / prob_under / prob_neutral regardless of class order
for want in ["over","under","neutral"]:
    if f"prob_{want}" not in out.columns:
        # find where that class is and rename
        if want in class_order:
            idx = class_order.index(want)
            out[f"prob_{want}"] = proba_test[:, idx]
        else:
            out[f"prob_{want}"] = np.nan

# Rank by probability of overperformance
out_ranked = out.sort_values("prob_over", ascending=False)
print("\nTop 15 RBs by probability of OVER-performing (Test 2024):")
print(out_ranked.head(15))

# ===== Optional: save probabilities =====
# out_ranked.to_csv(r"C:\Users\idhan\Downloads\... \rb_overprob_test2024.csv", index=False)

# ===== Also show likely busts (prob_under) =====
print("\nTop 15 RBs by probability of UNDER-performing (Test 2024):")
print(out.sort_values("prob_under", ascending=False).head(15))

joblib.dump(rf,r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\rb_model_random_forest_classification.pkl")