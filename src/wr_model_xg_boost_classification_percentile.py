import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---------- Load ----------
wr = pd.read_csv(
    r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_wr_vet_data.csv"
)

# ---------- Feature set (same style as TEs, trimmed to numeric & present) ----------
feature_candidates = [
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
# keep numeric columns that exist
features = [c for c in feature_candidates if c in wr.columns and pd.api.types.is_numeric_dtype(wr[c])]

# ---------- Build label with TRAIN percentiles ----------
wr["ppg_diff"] = wr["AVG"] - wr["expected_ppr_pg_curr_hist"]

# ensure merge_year is numeric
wr["merge_year"] = pd.to_numeric(wr["merge_year"], errors="coerce")

# Drop rows that can't be used
wr = wr.dropna(subset=["ppg_diff","merge_year"]).copy()

# Splits aligned with your other positions
train_mask = wr["merge_year"].between(2016, 2020)
val_mask   = wr["merge_year"].between(2021, 2023)
test_mask  = wr["merge_year"].eq(2024)

# Compute percentile thresholds **on TRAIN only**
q_low  = np.nanpercentile(wr.loc[train_mask, "ppg_diff"], 20)
q_high = np.nanpercentile(wr.loc[train_mask, "ppg_diff"], 80)

def label_from_diff(x, lo=q_low, hi=q_high):
    if pd.isna(x): 
        return np.nan
    if x <= lo:      return "under"
    if x >= hi:      return "over"
    return "neutral"

wr["label"] = wr["ppg_diff"].apply(label_from_diff)

# Filter to rows that have a label
wr = wr.dropna(subset=["label"]).copy()

# ---------- Prepare matrices ----------
X_all = wr[features]
y_all = wr["label"]

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_val,   y_val   = X_all[val_mask],   y_all[val_mask]
X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

# Impute missing values (median)
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_val_i   = imp.transform(X_val)
X_test_i  = imp.transform(X_test)

# Encode labels -> {under:0, neutral:1, over:2} (fixed mapping for readability)
label_map = {"under":0, "neutral":1, "over":2}
inv_label = {v:k for k,v in label_map.items()}

y_train_i = y_train.map(label_map).astype(int).values
y_val_i   = y_val.map(label_map).astype(int).values
y_test_i  = y_test.map(label_map).astype(int).values

# ---------- XGBoost model ----------
clf = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=800,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=42,
    tree_method="hist",      # fast, CPU-friendly
    eval_metric="mlogloss"
)

clf.fit(
    X_train_i, y_train_i,
    eval_set=[(X_val_i, y_val_i)],
    verbose=False,
)

# ---------- Evaluate ----------
from sklearn.metrics import classification_report, confusion_matrix

def pretty_report(X_i, y_true_i, split_name):
    y_pred_i = clf.predict(X_i)
    print(f"\n{split_name} Performance:")
    print(classification_report(y_true_i, y_pred_i, target_names=["under","neutral","over"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_i, y_pred_i))

pretty_report(X_val_i,  y_val_i,  "Validation")
pretty_report(X_test_i, y_test_i, "Test")

# ---------- Probabilities and top lists (Test / 2024) ----------
probs_test = clf.predict_proba(X_test_i)  # shape (n,3) aligned with [0:under,1:neutral,2:over]
prob_df = pd.DataFrame({
    "prob_under": probs_test[:, 0],
    "prob_neutral": probs_test[:, 1],
    "prob_over": probs_test[:, 2],
}, index=y_test.index)

# Attach identifiers
show_cols = ["Player_fixed","display_name","FantPos","merge_year"]
for c in show_cols:
    if c in wr.columns:
        prob_df[c] = wr.loc[y_test.index, c].values

# Top 15 by OVER probability
top_over = prob_df.sort_values("prob_over", ascending=False).head(15)
print("\nTop 15 WRs by probability of OVER-performing (Test 2024):")
print(top_over[show_cols + ["prob_neutral","prob_over","prob_under"]])

# Top 15 by UNDER probability
top_under = prob_df.sort_values("prob_under", ascending=False).head(15)
print("\nTop 15 WRs by probability of UNDER-performing (Test 2024):")
print(top_under[show_cols + ["prob_neutral","prob_over","prob_under"]])

# ---------- Quick class balance sanity ----------
print("\nClass counts (train):")
print(pd.Series(y_train_i).map(inv_label).value_counts())
print("\nClass counts (val):")
print(pd.Series(y_val_i).map(inv_label).value_counts())
print("\nClass counts (test):")
print(pd.Series(y_test_i).map(inv_label).value_counts())

joblib.dump(clf,r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\wr_model_xg_boost_classification_percentile.pkl")