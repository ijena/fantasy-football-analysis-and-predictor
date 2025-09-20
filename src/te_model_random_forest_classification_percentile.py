import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
# ===== Load =====
te = pd.read_csv(
    r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_te_vet_data.csv"
)

# ===== Feature list =====
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

# Keep only existing numeric columns
features = [c for c in features if c in te.columns and pd.api.types.is_numeric_dtype(te[c])]

# --- Robust column pickers ---
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

ppr_col   = pick_col(te, ["fantasy_points_ppr", "Fantasy_PPR"])
games_col = pick_col(te, ["Games_G", "games", "G"])
exp_pg    = pick_col(te, ["expected_ppr_pg_curr_hist", "xpected_ppr_pg_curr_hist"])  # typo-safe
year_col  = pick_col(te, ["merge_year", "year"])
id_cols   = [c for c in ["Player_fixed","display_name","FantPos", year_col] if c in te.columns]

missing = [name for name, col in {
    "fantasy_points_ppr": ppr_col,
    "Games_G/games": games_col,
    "expected_ppr_pg_curr_hist": exp_pg,
    "merge_year/year": year_col
}.items() if col is None]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ===== Label construction (percentiles from TRAIN only) =====
te["ppg_ppr"]   = pd.to_numeric(te[ppr_col], errors="coerce") / pd.to_numeric(te[games_col], errors="coerce")
te["perf_diff"] = te["ppg_ppr"] - pd.to_numeric(te[exp_pg], errors="coerce")
te[year_col]    = pd.to_numeric(te[year_col], errors="coerce")

# Drop rows without target or year
te = te.dropna(subset=["perf_diff", year_col]).copy()

train_mask = te[year_col].between(2016, 2020)
val_mask   = te[year_col].between(2021, 2023)
test_mask  = te[year_col].eq(2024)

train_diffs = te.loc[train_mask, "perf_diff"].replace([np.inf, -np.inf], np.nan).dropna()
low_th  = np.percentile(train_diffs, 20)
high_th = np.percentile(train_diffs, 80)
print(f"Label thresholds (from TRAIN): under ≤ {low_th:.2f}, neutral between, over ≥ {high_th:.2f}")

def label_perf(x: float) -> str:
    if x <= low_th:  return "under"
    if x >= high_th: return "over"
    return "neutral"

te["perf_class"] = te["perf_diff"].apply(label_perf)

# ===== Split matrices =====
X_train, y_train = te.loc[train_mask, features], te.loc[train_mask, "perf_class"]
X_val,   y_val   = te.loc[val_mask,   features], te.loc[val_mask,   "perf_class"]
X_test,  y_test  = te.loc[test_mask,  features], te.loc[test_mask,  "perf_class"]

# ===== Impute =====
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_val_i   = imp.transform(X_val)
X_test_i  = imp.transform(X_test)

# ===== Class weighting (counter class imbalance) =====
cls_counts = y_train.value_counts()
avg = cls_counts.mean()
weights = {cls: (avg / cnt) for cls, cnt in cls_counts.items()}
sample_w = y_train.map(weights).values

# ===== Model =====
rf = RandomForestClassifier(
    n_estimators=600,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_leaf=2,
    class_weight=None  # using custom sample weights
)
rf.fit(X_train_i, y_train, sample_weight=sample_w)

# ===== Evaluate =====
print("\nValidation Performance:\n", classification_report(y_val, rf.predict(X_val_i), labels=["under","neutral","over"]))
print("Confusion Matrix (VAL):\n", confusion_matrix(y_val, rf.predict(X_val_i), labels=["under","neutral","over"]))

print("\nTest Performance:\n", classification_report(y_test, rf.predict(X_test_i), labels=["under","neutral","over"]))
print("Confusion Matrix (TEST):\n", confusion_matrix(y_test, rf.predict(X_test_i), labels=["under","neutral","over"]))

# ===== Probability tables for 2024 =====
if hasattr(rf, "predict_proba") and test_mask.any():
    proba = rf.predict_proba(X_test_i)  # columns follow rf.classes_
    class_order = list(rf.classes_)
    prob_df = pd.DataFrame(proba, columns=[f"prob_{c}" for c in class_order], index=X_test.index)

    out = pd.concat([te.loc[test_mask, id_cols], prob_df], axis=1)

    # Helper for missing class columns (if a class wasn’t present in TRAIN)
    def c(df, name):
        return name if name in df.columns else None

    col_over  = c(out, "prob_over")
    col_under = c(out, "prob_under")

    if col_over:
        print("\nTop 15 TEs by probability of OVER-performing (Test 2024):")
        print(out.sort_values(col_over, ascending=False).head(15))
    else:
        print("\n[Note] No 'over' class learned in training; cannot rank by prob_over.")

    if col_under:
        print("\nTop 15 TEs by probability of UNDER-performing (Test 2024):")
        print(out.sort_values(col_under, ascending=False).head(15))
    else:
        print("\n[Note] No 'under' class learned in training; cannot rank by prob_under.")
else:
    print("\nProbabilities unavailable (no predict_proba or empty test split).")

joblib.dump(rf,r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\te_model_random_forest_classification_percentile.pkl")