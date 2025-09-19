# XGBoost TE classifier with 20/80 percentile thresholds
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

# ==== Load ====
te = pd.read_csv(
    r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_te_vet_data.csv"
)

# ==== Safety: compute per-game + label target ====
# ppg actual = Fantasy_PPR / Games_G (guard divide by 0)
te["ppg_Fantasy_PPR"] = np.where(te["Games_G"].fillna(0) > 0,
                                 te["fantasy_points_ppr"] / te["Games_G"],
                                 np.nan)

# expected from your historical-ADP mapping (already in file)
exp_col = "expected_ppr_pg_curr_hist"
if exp_col not in te.columns:
    raise ValueError(f"Expected column '{exp_col}' missing in TE file.")

te["perf_pg"] = te["ppg_Fantasy_PPR"] - te[exp_col]

# Year-based split (train: 2016–2020, val: 2021–2023, test: 2024)
if "merge_year" not in te.columns:
    raise ValueError("Column 'merge_year' not found. Make sure it exists in the TE dataset.")

train_mask = te["merge_year"].between(2016, 2020)
val_mask   = te["merge_year"].between(2021, 2023)
test_mask  = te["merge_year"].eq(2024)

# ==== Build thresholds from TRAIN only (20/80) ====
perf_train = te.loc[train_mask, "perf_pg"].dropna()
under_th = np.nanpercentile(perf_train, 20)  # <= under
over_th  = np.nanpercentile(perf_train, 80)  # >= over
print(f"Label thresholds from TRAIN: under ≤ {under_th:.2f}, neutral between, over ≥ {over_th:.2f}")

def to_class(x):
    if pd.isna(x):
        return np.nan
    if x <= under_th:
        return "under"
    elif x >= over_th:
        return "over"
    else:
        return "neutral"

te["perf_class"] = te["perf_pg"].apply(to_class)

# Drop rows missing label or year
te = te.dropna(subset=["perf_class", "merge_year"]).copy()

# ==== Feature set ====
features = [
    "age","g","gs","tgt","rec","yds","td","x1d","ybc","ybc_r","yac","yac_r","adot",
    "brk_tkl","rec_br","drop","drop_percent","int","rat","avg_cushion","avg_separation",
    "avg_intended_air_yards","percent_share_of_intended_air_yards","catch_percentage",
    "avg_expected_yac","avg_yac_above_expectation","receiving_fumbles",
    "receiving_fumbles_lost","receiving_air_yards","receiving_epa","receiving_2pt_conversions",
    "racr","target_share","air_yards_share","fantasy_points","fantasy_points_ppr",
    "games","tgt_sh","ay_sh","yac_sh","wopr_y","ry_sh","rtd_sh","rfd_sh","dom",
    "w8dom","yptmpa","ppr_sh","height","weight","draft_round","draft_pick",
    "offense_snaps","team_snaps","offense_snap_percentage","AVG","adp_percentile",
    "adp_percentile_pos", "expected_ppr_pg_curr_hist","Games_G"
]

# keep only numeric features that exist
features = [c for c in features if c in te.columns and pd.api.types.is_numeric_dtype(te[c])]
X = te[features].copy()
y = te["perf_class"].copy()

# ==== Split ====
X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

# ==== Impute ====
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_val_i   = imp.transform(X_val)
X_test_i  = imp.transform(X_test)

# ==== Encode labels to [0,1,2] ====
le = LabelEncoder()
le.fit(["under","neutral","over"])  # ensure consistent class order
y_train_i = le.transform(y_train)
y_val_i   = le.transform(y_val)
y_test_i  = le.transform(y_test)

# ==== Class weights (handle imbalance) ====
sample_w = compute_sample_weight(class_weight="balanced", y=y_train_i)

# ==== XGBoost model ====
clf = XGBClassifier(
    n_estimators=800,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss",
)
clf.fit(X_train_i, y_train_i, sample_weight=sample_w)

# ==== Evaluate ====
def show_results(split_name, X_i, y_true_i):
    y_pred_i = clf.predict(X_i)
    print(f"\n{split_name} Performance:")
    print(classification_report(le.inverse_transform(y_true_i),
                                le.inverse_transform(y_pred_i)))
    print("Confusion Matrix:")
    print(confusion_matrix(le.inverse_transform(y_true_i),
                           le.inverse_transform(y_pred_i)))

show_results("Validation", X_val_i, y_val_i)
show_results("Test",       X_test_i, y_test_i)

# ==== Probability tables (Test 2024) ====
proba_test = clf.predict_proba(X_test_i)
proba_df = pd.DataFrame(proba_test, columns=le.inverse_transform([0,1,2]))
# columns order is by label encoder mapping
proba_df = proba_df.rename(columns={"under":"prob_under","neutral":"prob_neutral","over":"prob_over"})

cols_to_show = ["Player_fixed","display_name","FantPos","merge_year"]
cols_to_show = [c for c in cols_to_show if c in te.columns]
out_test = pd.concat([te.loc[test_mask, cols_to_show].reset_index(drop=True), proba_df], axis=1)

# Top likely over-performers
top_over = out_test.sort_values("prob_over", ascending=False).head(15)
print("\nTop 15 TEs by probability of OVER-performing (Test 2024):")
print(top_over)

# Top likely under-performers
top_under = out_test.sort_values("prob_under", ascending=False).head(15)
print("\nTop 15 TEs by probability of UNDER-performing (Test 2024):")
print(top_under)
