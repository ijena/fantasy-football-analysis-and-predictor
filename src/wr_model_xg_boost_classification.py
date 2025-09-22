import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

# ------------------ Load Data ------------------
wr = pd.read_csv(
    r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_wr_vet_data.csv"
)

# ------------------ Features ------------------
raw_features = [
    "age","g","gs","tgt","rec","yds","td","x1d","ybc","ybc_r","yac","yac_r","adot",
    "brk_tkl","rec_br","drop","drop_percent","int","rat",
    "avg_cushion","avg_separation","avg_intended_air_yards","percent_share_of_intended_air_yards",
    "receptions_x","targets_x","avg_yac","avg_expected_yac","avg_yac_above_expectation",
    "receiving_fumbles","receiving_fumbles_lost","receiving_air_yards","receiving_epa","receiving_2pt_conversions",
    "racr","target_share","air_yards_share","wopr_x","fantasy_points",
    "games","tgt_sh","ay_sh","yac_sh","wopr_y","ry_sh","rtd_sh","rfd_sh","dom","w8dom","yptmpa","ppr_sh",
    "height","weight","draft_round","draft_pick",
    "offense_snaps","team_snaps","AVG","adp_percentile",
    "expected_ppr_pg_curr_hist","Games_G"
]
# keep only columns that exist and are numeric
feature_cols = [c for c in raw_features if c in wr.columns and pd.api.types.is_numeric_dtype(wr[c])]

# ------------------ Label Creation ------------------
# difference between actual and expected
wr["ppg_diff"] = wr["AVG"] - wr["expected_ppr_pg_curr_hist"]

wr = wr.dropna(subset=["ppg_diff"])

# Asymmetric thresholds
UNDER_TH = -2.0
OVER_TH  = 5.0

def label_func(x):
    if x <= UNDER_TH:
        return "under"
    elif x >= OVER_TH:
        return "over"
    else:
        return "neutral"

wr["label"] = wr["ppg_diff"].apply(label_func)

# ------------------ Year splits ------------------
wr["merge_year"] = pd.to_numeric(wr["merge_year"], errors="coerce")
mask_valid_year = wr["merge_year"].notna()

# train: 2016-2020, val: 2021-2023, test: 2024
train_mask = mask_valid_year & wr["merge_year"].between(2016, 2020)
val_mask   = mask_valid_year & wr["merge_year"].between(2021, 2023)
test_mask  = mask_valid_year & (wr["merge_year"] == 2024)

train = wr.loc[train_mask].copy()
val   = wr.loc[val_mask].copy()
test  = wr.loc[test_mask].copy()

# if any split is empty, fail early
for name, df in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
    if df.empty:
        raise ValueError(f"{name} split is empty. Check merge_year values in your data.")

# ------------------ Prepare Data ------------------
X_train, y_train = train[feature_cols], train["label"]
X_val,   y_val   = val[feature_cols],   val["label"]
X_test,  y_test  = test[feature_cols],  test["label"]

# ensure consistent numeric encoding for labels
CLASS_NAMES = ["neutral", "over", "under"]           # fixed order
label_to_int = {lbl:i for i,lbl in enumerate(CLASS_NAMES)}

def encode_labels(y_series):
    # map known labels; drop rows that map to NaN (shouldn't happen, but safe)
    y_enc = y_series.map(label_to_int)
    if y_enc.isna().any():
        bad = y_series[y_enc.isna()].unique()
        raise ValueError(f"Unknown labels encountered: {bad}")
    return y_enc.astype(int)

y_train_enc = encode_labels(y_train)
y_val_enc   = encode_labels(y_val)
y_test_enc  = encode_labels(y_test)

# ------------------ Class counts ------------------
print("Class counts (train):")
print(y_train.value_counts())
print("\nClass counts (val):")
print(y_val.value_counts())
print("\nClass counts (test):")
print(y_test.value_counts())

# ------------------ XGBoost Model ------------------
# XGBoost can handle np.nan directly; no imputation required here.
xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    tree_method="hist",       # fast default; use "gpu_hist" if you have a GPU
    max_depth=6,
    n_estimators=600,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

xgb.fit(
    X_train, y_train_enc,
    eval_set=[(X_val, y_val_enc)],
    verbose=False
)

# ------------------ Reporting ------------------
ALL_LABELS = [0,1,2]  # corresponds to CLASS_NAMES order above

def report(split, y_true_enc, y_pred_enc):
    print(f"\n{split} Performance:")
    print(
        classification_report(
            y_true_enc,
            y_pred_enc,
            labels=ALL_LABELS,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_enc, y_pred_enc, labels=ALL_LABELS))

# Validation
y_val_pred = xgb.predict(X_val)
report("Validation", y_val_enc, y_val_pred)

# Test
y_test_pred = xgb.predict(X_test)
report("Test", y_test_enc, y_test_pred)

# ------------------ Probabilities & Rankings (Test 2024) ------------------
y_test_proba = xgb.predict_proba(X_test)
proba_df = pd.DataFrame(y_test_proba, columns=[f"prob_{c}" for c in CLASS_NAMES])

results = pd.concat([test.reset_index(drop=True), proba_df], axis=1)

print("\nTop 15 WRs by probability of OVER-performing (Test 2024):")
print(
    results.sort_values("prob_over", ascending=False)
           .head(15)[["Player_fixed","display_name","FantPos","merge_year",
                      "prob_neutral","prob_over","prob_under"]]
)

print("\nTop 15 WRs by probability of UNDER-performing (Test 2024):")
print(
    results.sort_values("prob_under", ascending=False)
           .head(15)[["Player_fixed","display_name","FantPos","merge_year",
                      "prob_neutral","prob_over","prob_under"]]
)

joblib.dump(xgb,r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\wr_model_xg_boost_classification.pkl")