import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ===== Load data =====
qb_model_data = pd.read_csv(
    r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_qb_vet_data.csv"
)

# ----- Labels -----
qb_model_data["ppg_Fantasy_PPR"] = qb_model_data["Fantasy_PPR"] / qb_model_data["Games_G"]
qb_model_data["per_game_perf_rel_expectations"] = (
    qb_model_data["ppg_Fantasy_PPR"] - qb_model_data["expected_ppr_pg_curr_hist"]
)

# Classification labels
def label_from_perf(x, th=2.0):
    if x >= th:
        return "over"
    elif x <= -th:
        return "under"
    else:
        return "neutral"

qb_model_data["perf_class"] = qb_model_data["per_game_perf_rel_expectations"].apply(label_from_perf)

# Drop NaNs
qb_model_data = qb_model_data.dropna(subset=["perf_class", "merge_year"])

# ===== Features =====
features = ["rank","qbr_total","pts_added","qb_plays","epa_total","pass","run",
            "exp_sack","penalty","qbr_raw","sack","pass_attempts","throwaways","spikes",
            "drops","drop_pct","bad_throws","bad_throw_pct","pocket_time","times_blitzed",
            "times_hurried","times_hit","times_pressured","pressure_pct","batted_balls",
            "on_tgt_throws","on_tgt_pct","rpo_plays","rpo_yards","rpo_pass_att","rpo_pass_yards",
            "rpo_rush_att","rpo_rush_yards","pa_pass_att","pa_pass_yards","avg_time_to_throw",
            "avg_completed_air_yards","avg_intended_air_yards","avg_air_yards_differential",
            "aggressiveness","max_completed_air_distance","avg_air_yards_to_sticks","pass_yards",
            "pass_touchdowns","interceptions_x","passer_rating","completion_percentage",
            "expected_completion_percentage","completion_percentage_above_expectation",
            "avg_air_distance","max_air_distance","sacks","sack_yards","sack_fumbles",
            "sack_fumbles_lost","passing_air_yards","passing_yards_after_catch","passing_epa",
            "passing_2pt_conversions","pacr","dakota","carries","rushing_yards","rushing_tds",
            "rushing_fumbles_x","rushing_fumbles_lost_x","rushing_first_downs","rushing_epa_x",
            "rushing_2pt_conversions_x","fantasy_points_x","games_x","ppr_sh","height_x","weight_x",
            "college_name_x","college_conference_x","draft_round_x","draft_pick_x","age","gs","ybc",
            "ybc_att","yac","yac_att","brk_tkl","att_br","offense_snaps","team_snaps","AVG",
            "adp_percentile","adp_percentile_pos","expected_ppr_pg_curr_hist","Games_G"]

# keep only numeric cols that exist
features = [c for c in features if c in qb_model_data.columns and pd.api.types.is_numeric_dtype(qb_model_data[c])]

X_all = qb_model_data[features]
y_all = qb_model_data["perf_class"]

# ===== Split by year =====
train = qb_model_data["merge_year"].between(2016, 2020)
val   = qb_model_data["merge_year"].between(2021, 2023)
test  = qb_model_data["merge_year"].eq(2024)

X_train, y_train = X_all[train], y_all[train]
X_val, y_val     = X_all[val], y_all[val]
X_test, y_test   = X_all[test], y_all[test]

# ===== Impute =====
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_val_i   = imp.transform(X_val)
X_test_i  = imp.transform(X_test)

# ===== Model =====
clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced")
clf.fit(X_train_i, y_train)

# ===== Evaluate =====
print("Validation Performance:\n", classification_report(y_val, clf.predict(X_val_i)))
print("Test Performance:\n", classification_report(y_test, clf.predict(X_test_i)))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, clf.predict(X_test_i)))

# Class balance check
print("\nClass counts (train):\n", y_train.value_counts())
print("\nClass counts (val):\n", y_val.value_counts())
print("\nClass counts (test):\n", y_test.value_counts())
