import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
# ===== Load RB dataset =====
rb_model_data = pd.read_csv(
    r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_rb_vet_data.csv"
)

# ----- Labels -----
rb_model_data["ppg_Fantasy_PPR"] = rb_model_data["Fantasy_PPR"] / rb_model_data["Games_G"]
rb_model_data["perf_rel_exp"] = (
    rb_model_data["ppg_Fantasy_PPR"] - rb_model_data["expected_ppr_pg_curr_hist"]
)

rb_model_data = rb_model_data.dropna(subset=["perf_rel_exp"])
# Compute thresholds based on TRAIN split only (to avoid leakage)
train_mask = rb_model_data["merge_year"].between(2016, 2020)
train_perf = rb_model_data.loc[train_mask, "perf_rel_exp"].dropna()

low_thr = np.percentile(train_perf, 30)   # bottom 30% = underperform
high_thr = np.percentile(train_perf, 70)  # top 30% = overperform

print(f"Label thresholds from TRAIN: under ≤ {low_thr:.2f}, "
      f"neutral between, over ≥ {high_thr:.2f}")

# Create labels
def label_from_percentile(x):
    if x <= low_thr:
        return "under"
    elif x >= high_thr:
        return "over"
    else:
        return "neutral"

rb_model_data["perf_class"] = rb_model_data["perf_rel_exp"].apply(label_from_percentile)

# Drop NaNs
rb_model_data = rb_model_data.dropna(subset=["perf_class", "merge_year"])

# ===== Features =====
features = ["att","yds_x","td_x","x1d_x","ybc_x","ybc_att","yac_x","yac_att","brk_tkl_x",
           "att_br","efficiency","percent_attempts_gte_eight_defenders",
           "expected_rush_yards","rush_yards_over_expected",
           "rush_yards_over_expected_per_att","rush_pct_over_expected","rushing_fumbles",
           "rushing_fumbles_lost","rushing_epa","rushing_2pt_conversions","fantasy_points_x",
           "fantasy_points_ppr_x","games_x","height_x","weight_x","draft_round_x",	
           "draft_pick_x","tgt","yds_y","td_y","x1d_y","ybc_y","ybc_r","yac_y","yac_r",
           "adot","brk_tkl_y","rec_br","drop","drop_percent","int","rat","receiving_fumbles",
           "receiving_fumbles_lost","receiving_air_yards","receiving_epa","receiving_2pt_conversions",
           "racr","target_share","air_yards_share","wopr_x","tgt_sh","ay_sh","yac_sh",	
           "wopr_y","ry_sh","rtd_sh","rfd_sh","dom","w8dom","yptmpa","ppr_sh","offense_snaps",
           "team_snaps","AVG","adp_percentile","adp_percentile_pos","expected_ppr_pg_curr_hist",
           "Games_G"]

# keep only numeric cols that exist
features = [c for c in features if c in rb_model_data.columns and pd.api.types.is_numeric_dtype(rb_model_data[c])]

X_all = rb_model_data[features]
y_all = rb_model_data["perf_class"]

# ===== Split =====
train = rb_model_data["merge_year"].between(2016, 2020)
val   = rb_model_data["merge_year"].between(2021, 2023)
test  = rb_model_data["merge_year"].eq(2024)

X_train, y_train = X_all[train], y_all[train]
X_val, y_val     = X_all[val], y_all[val]
X_test, y_test   = X_all[test], y_all[test]

# ===== Impute =====
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_val_i   = imp.transform(X_val)
X_test_i  = imp.transform(X_test)

# ===== Model =====
clf = RandomForestClassifier(
    n_estimators=400, 
    random_state=42, 
    n_jobs=-1, 
    class_weight="balanced"
)
clf.fit(X_train_i, y_train)

# ===== Evaluate =====
print("\nValidation Performance:\n", classification_report(y_val, clf.predict(X_val_i)))
print("Test Performance:\n", classification_report(y_test, clf.predict(X_test_i)))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, clf.predict(X_test_i)))

# Probabilities
probs = clf.predict_proba(X_test_i)
prob_df = pd.DataFrame(probs, columns=clf.classes_)
prob_df["Player_fixed"] = rb_model_data.loc[test, "Player_fixed"].values
prob_df["display_name"] = rb_model_data.loc[test, "display_name"].values

print("\nTop 15 RBs by probability of OVER-performing:")
print(prob_df.sort_values("over", ascending=False).head(15))

joblib.dump(clf,r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\rb_model_random_forest_classification_percentile.pkl")