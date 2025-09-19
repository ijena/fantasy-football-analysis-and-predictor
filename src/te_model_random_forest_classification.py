import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ===== Config =====
POINT_THR = 2.0  # +/- PPR per game to define over/under

# ===== Load TE data =====
te = pd.read_csv(
    r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_te_vet_data.csv"
)

# ===== Feature list (as provided) =====
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

# --- helper to robustly pick columns if names differ slightly ---
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

ppr_col   = pick_col(te, ["fantasy_points_ppr", "Fantasy_PPR", "Fantasy_PPR_x"])
games_col = pick_col(te, ["Games_G", "games", "G"])
exp_pg    = pick_col(te, ["expected_ppr_pg_curr_hist", "xpected_ppr_pg_curr_hist"])
year_col  = pick_col(te, ["merge_year", "year"])  # we expect merge_year from your pipeline

missing = [name for name, col in {
    "fantasy_points_ppr": ppr_col,
    "Games_G/games": games_col,
    "expected_ppr_pg_curr_hist": exp_pg,
    "merge_year/year": year_col
}.items() if col is None]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ===== Targets: delta vs expected (per game) =====
te["ppg_ppr"] = pd.to_numeric(te[ppr_col], errors="coerce") / pd.to_numeric(te[games_col], errors="coerce")
te["delta_ppg"] = te["ppg_ppr"] - pd.to_numeric(te[exp_pg], errors="coerce")
te[year_col] = pd.to_numeric(te[year_col], errors="coerce")

# Remove rows without target or year
te = te.dropna(subset=["delta_ppg", year_col]).copy()

# ===== Labeling (fixed threshold) =====
LOW_THR  = -POINT_THR
HIGH_THR =  POINT_THR

def label_from_delta(x: float) -> str:
    if x <= LOW_THR:  return "under"
    if x >= HIGH_THR: return "over"
    return "neutral"

te["perf_class"] = te["delta_ppg"].apply(label_from_delta)

# ===== Build feature matrix =====
# keep only numeric columns that exist
feature_cols = [c for c in features if (c in te.columns and pd.api.types.is_numeric_dtype(te[c]))]
X_all = te[feature_cols]
y_all = te["perf_class"]

# ===== Time-based splits =====
train_mask = te[year_col].between(2016, 2020)
val_mask   = te[year_col].between(2021, 2023)
test_mask  = te[year_col].eq(2024)

X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
X_val,   y_val   = X_all.loc[val_mask],   y_all.loc[val_mask]
X_test,  y_test  = X_all.loc[test_mask],  y_all.loc[test_mask]

# ===== Impute =====
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_val_i   = imp.transform(X_val)
X_test_i  = imp.transform(X_test)

# ===== Class weights (to counter neutral dominance) =====
# Compute simple inverse-frequency weights on TRAIN
cls_counts = y_train.value_counts()
avg = cls_counts.mean()
weights = {cls: (avg / cnt) for cls, cnt in cls_counts.items()}
sample_w = y_train.map(weights).values

# ===== Train RF =====
rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight=None  # using per-sample weights instead
)
rf.fit(X_train_i, y_train, sample_weight=sample_w)

# ===== Evaluate =====
print("Validation Performance:\n", classification_report(y_val, rf.predict(X_val_i), labels=["under","neutral","over"]))
print("Confusion Matrix (VAL):\n", confusion_matrix(y_val, rf.predict(X_val_i), labels=["under","neutral","over"]))

print("\nTest Performance:\n", classification_report(y_test, rf.predict(X_test_i), labels=["under","neutral","over"]))
print("Confusion Matrix (TEST):\n", confusion_matrix(y_test, rf.predict(X_test_i), labels=["under","neutral","over"]))

# ===== Probability tables (Test 2024) =====
if hasattr(rf, "predict_proba"):
    proba = rf.predict_proba(X_test_i)  # order follows rf.classes_
    # Ensure columns align to ["under","neutral","over"]
    classes = list(rf.classes_)
    prob_df = pd.DataFrame(proba, columns=[f"prob_{c}" for c in classes], index=X_test.index)
    # Add ID columns if present
    id_cols = [c for c in ["Player_fixed","display_name","FantPos",year_col] if c in te.columns]
    out = pd.concat([te.loc[test_mask, id_cols], prob_df], axis=1)

    # Sort helper: handle missing columns if any class absent in train
    def get_col_safe(df, name):
        return name if name in df.columns else None

    col_over  = get_col_safe(out, "prob_over")
    col_under = get_col_safe(out, "prob_under")

    if col_over:
        print("\nTop 15 TEs by probability of OVER-performing (Test 2024):")
        print(out.sort_values(col_over, ascending=False).head(15))
    else:
        print("\n[Note] 'over' class missing in training split; no over-probabilities to show.")

    if col_under:
        print("\nTop 15 TEs by probability of UNDER-performing (Test 2024):")
        print(out.sort_values(col_under, ascending=False).head(15))
    else:
        print("\n[Note] 'under' class missing in training split; no under-probabilities to show.")
else:
    print("\nModel does not support predict_proba; probabilities unavailable.")
