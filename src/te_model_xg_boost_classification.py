import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
# ===== Load TE data =====
te = pd.read_csv(
    r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_te_vet_data.csv"
)

# ===== Features =====
features = ["age","g","gs","tgt","rec","yds","td","x1d","ybc","ybc_r","yac","yac_r","adot",
            "brk_tkl","rec_br","drop","drop_percent","int","rat","avg_cushion","avg_separation",
            "avg_intended_air_yards","percent_share_of_intended_air_yards","catch_percentage",
            "avg_expected_yac","avg_yac_above_expectation","receiving_fumbles",
            "receiving_fumbles_lost","receiving_air_yards","receiving_epa","receiving_2pt_conversions",
            "racr","target_share","air_yards_share","fantasy_points","fantasy_points_ppr",	
            "games","tgt_sh","ay_sh","yac_sh","wopr_y","ry_sh","rtd_sh","rfd_sh","dom",	
            "w8dom","yptmpa","ppr_sh","height","weight","draft_round","draft_pick",
            "offense_snaps","team_snaps","offense_snap_percentage","AVG","adp_percentile",
            "adp_percentile_pos","expected_ppr_pg_curr_hist","Games_G"]

# keep only numeric cols
features = [c for c in features if c in te.columns and pd.api.types.is_numeric_dtype(te[c])]

# ===== Labels =====
te["ppg_Fantasy_PPR"] = te["fantasy_points_ppr"] / te["Games_G"]
te["perf_rel_exp"] = te["ppg_Fantasy_PPR"] - te["expected_ppr_pg_curr_hist"]

# Threshold classification
def label_from_perf(x, th=2.0):
    if x >= th:
        return "over"
    elif x <= -th:
        return "under"
    else:
        return "neutral"

te["perf_class"] = te["perf_rel_exp"].apply(label_from_perf)

# Drop missing
te = te.dropna(subset=["perf_class", "merge_year"])

# ===== Split =====
train = te["merge_year"].between(2016, 2020)
val   = te["merge_year"].between(2021, 2023)
test  = te["merge_year"].eq(2024)

X_all = te[features]
y_all = te["perf_class"]

X_train, y_train = X_all[train], y_all[train]
X_val, y_val     = X_all[val], y_all[val]
X_test, y_test   = X_all[test], y_all[test]

# ===== Impute =====
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_val_i   = imp.transform(X_val)
X_test_i  = imp.transform(X_test)

# ===== XGBoost model =====
clf = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="multi:softprob",  # multi-class with probabilities
    eval_metric="mlogloss"
)

# Encode labels numerically for XGBoost
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)

clf.fit(X_train_i, y_train_enc)

# ===== Evaluate =====
print("Validation Performance:\n", classification_report(y_val_enc, clf.predict(X_val_i), target_names=le.classes_))
print("Confusion Matrix (VAL):\n", confusion_matrix(y_val_enc, clf.predict(X_val_i)))

print("\nTest Performance:\n", classification_report(y_test_enc, clf.predict(X_test_i), target_names=le.classes_))
print("Confusion Matrix (TEST):\n", confusion_matrix(y_test_enc, clf.predict(X_test_i)))

# ===== Probability rankings =====
probs = clf.predict_proba(X_test_i)
probs_df = pd.DataFrame(probs, columns=[f"prob_{cls}" for cls in le.classes_])
probs_df = pd.concat([te.loc[test, ["Player_fixed","display_name","FantPos","merge_year"]].reset_index(drop=True),
                      probs_df], axis=1)

print("\nTop 10 TEs by probability of OVER-performing (Test 2024):")
print(probs_df.sort_values("prob_over", ascending=False).head(10))

print("\nTop 10 TEs by probability of UNDER-performing (Test 2024):")
print(probs_df.sort_values("prob_under", ascending=False).head(10))

joblib.dump(clf,r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\te_model_xg_boost_classification.pkl")