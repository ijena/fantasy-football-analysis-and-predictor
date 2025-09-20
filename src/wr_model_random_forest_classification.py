import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---------------- Load WR Data ----------------
wr = pd.read_csv(
    r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_wr_vet_data.csv"
)

# ---------------- Detect actual PPR (season total) & games ----------------
# We'll try these in order; adjust if your file uses different names.
PPR_CANDIDATES = ["Fantasy_PPR", "fantasy_points_ppr", "fantasy_points_ppr_x", "fantasy_points"]
GAMES_CANDIDATES = ["Games_G", "games", "games_x", "G"]

def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

ppr_col = first_col(wr, PPR_CANDIDATES)
games_col = first_col(wr, GAMES_CANDIDATES)

if ppr_col is None:
    raise ValueError("Could not find a season total PPR column (tried: %s)" % PPR_CANDIDATES)
if games_col is None:
    raise ValueError("Could not find a games column (tried: %s)" % GAMES_CANDIDATES)
if "expected_ppr_pg_curr_hist" not in wr.columns:
    raise ValueError("expected_ppr_pg_curr_hist not found in WR dataset.")

# ---------------- Compute per-game actual and label (±2.0 PPG threshold) ----------------
wr["ppg_actual"] = wr[ppr_col] / wr[games_col].replace(0, np.nan)
wr["ppg_diff"] = wr["ppg_actual"] - wr["expected_ppr_pg_curr_hist"]

THRESH = 2.0  # over/under threshold in PPR per game

def label_perf(x, th=THRESH):
    if pd.isna(x):
        return np.nan
    if x >= th:
        return "over"
    elif x <= -th:
        return "under"
    else:
        return "neutral"

wr["label"] = wr["ppg_diff"].apply(label_perf)

# ---------------- Year-based split: train/val/test ----------------
# Train on 2016–2020, validate on 2021–2023, test on 2024 (what you'll ship for 2025 draft)
if "merge_year" not in wr.columns:
    raise ValueError("merge_year not found in WR dataset.")

wr["merge_year"] = pd.to_numeric(wr["merge_year"], errors="coerce")

valid_mask = wr["label"].notna() & wr["merge_year"].notna()
wr = wr.loc[valid_mask].copy()

train_mask = wr["merge_year"].between(2016, 2020)
val_mask   = wr["merge_year"].between(2021, 2023)
test_mask  = wr["merge_year"].eq(2024)

# ---------------- Feature set (start with your TE-like set, keep only numeric & existing) ----------------
raw_features = [
    "age","g","gs","tgt","rec","yds","td","x1d","ybc","ybc_r","yac","yac_r","adot",
    "brk_tkl","rec_br","drop","drop_percent","int","rat","avg_cushion","avg_separation",
    "avg_intended_air_yards","percent_share_of_intended_air_yards",
    "receptions_x","catch_percentage","targets_x","avg_yac","avg_expected_yac","avg_yac_above_expectation",
    "receiving_fumbles","receiving_fumbles_lost","receiving_air_yards","receiving_epa","receiving_2pt_conversions",
    "racr","target_share","air_yards_share","wopr_x","fantasy_points",
    "games","tgt_sh","ay_sh","yac_sh","wopr_y","ry_sh","rtd_sh","rfd_sh","dom",
    "w8dom","yptmpa","ppr_sh","height","weight","draft_round","draft_pick",
    "offense_snaps","team_snaps","AVG","adp_percentile",
    "expected_ppr_pg_curr_hist","Games_G"
]

# keep only columns that exist and are numeric
features = [c for c in raw_features if c in wr.columns and pd.api.types.is_numeric_dtype(wr[c])]
if len(features) == 0:
    raise ValueError("No numeric features found from the provided list.")

X_all = wr[features]
y_all = wr["label"].astype("category")  # 'over', 'under', 'neutral'

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_val,   y_val   = X_all[val_mask],   y_all[val_mask]
X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

# ---------------- Impute missing values ----------------
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_val_i   = imp.transform(X_val)
X_test_i  = imp.transform(X_test)

# ---------------- Random Forest (balanced) ----------------
clf = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"  # helps when 'over'/'under' are smaller than 'neutral'
)
clf.fit(X_train_i, y_train)

# ---------------- Evaluate ----------------
print("Validation Performance:")
print(classification_report(y_val, clf.predict(X_val_i)))
print("Confusion Matrix (VAL):")
print(confusion_matrix(y_val, clf.predict(X_val_i)))

print("\nTest Performance:")
print(classification_report(y_test, clf.predict(X_test_i)))
print("Confusion Matrix (TEST):")
print(confusion_matrix(y_test, clf.predict(X_test_i)))

# ---------------- Probabilities (TEST = 2024 only) ----------------
probs_test = clf.predict_proba(X_test_i)
classes = clf.classes_  # should be alphabetical: ['neutral','over','under'] typically

prob_df = pd.DataFrame(probs_test, columns=[f"prob_{cls}" for cls in classes], index=X_test.index)

# Bring back identifiers for 2024 rows
id_cols = [c for c in ["Player_fixed","display_name","FantPos","merge_year"] if c in wr.columns]
prob_df[id_cols] = wr.loc[X_test.index, id_cols]

# sort by over-perform probability if it exists
over_col = "prob_over" if "prob_over" in prob_df.columns else None
under_col = "prob_under" if "prob_under" in prob_df.columns else None
neutral_col = "prob_neutral" if "prob_neutral" in prob_df.columns else None

if over_col:
    top_over = prob_df.sort_values(over_col, ascending=False).head(15)
    print("\nTop 15 WRs by probability of OVER-performing (Test 2024):")
    keep_cols = id_cols + [c for c in [neutral_col, over_col, under_col] if c]
    print(top_over[keep_cols])

if under_col:
    top_under = prob_df.sort_values(under_col, ascending=False).head(15)
    print("\nTop 15 WRs by probability of UNDER-performing (Test 2024):")
    keep_cols = id_cols + [c for c in [neutral_col, over_col, under_col] if c]
    print(top_under[keep_cols])

# ---------------- Class balance sanity checks ----------------
print("\nClass counts (train):")
print(y_train.value_counts())
print("\nClass counts (val):")
print(y_val.value_counts())
print("\nClass counts (test):")
print(y_test.value_counts())

joblib.dump(clf,r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\wr_model_random_forest_classification.pkl")