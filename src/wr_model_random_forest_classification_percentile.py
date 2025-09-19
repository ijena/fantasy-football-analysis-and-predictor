import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# Load WR data
# =========================
wr = pd.read_csv(
    r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_wr_vet_data.csv"
)

# Defensive: ensure these exist
need = ["AVG", "expected_ppr_pg_curr_hist", "merge_year", "FantPos", "Player_fixed", "display_name"]
missing = [c for c in need if c not in wr.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# =========================
# Per-game difference (actual - expected)
# =========================
# If you already computed per-game actual elsewhere, great.
# Here we follow your earlier approach based on expected_ppr_pg_curr_hist vs actual-per-ADP proxy:
wr["ppg_diff"] = wr["AVG"] - wr["expected_ppr_pg_curr_hist"]

# =========================
# Label by YEAR percentiles (20/60/20) — computed separately within each year
# =========================
def label_year_percentiles(df: pd.DataFrame, year_col: str, diff_col: str, low_q=0.2, high_q=0.8):
    df = df.copy()
    labels = []
    for y, g in df.groupby(year_col):
        diffs = g[diff_col].dropna()
        if len(diffs) < 5:
            # Not enough samples: default everyone to neutral
            y_low, y_high = -np.inf, np.inf
        else:
            y_low = np.quantile(diffs, low_q)
            y_high = np.quantile(diffs, high_q)
        # assign
        lab = np.where(g[diff_col] >= y_high, "over",
               np.where(g[diff_col] <= y_low, "under", "neutral"))
        labels.append(pd.Series(lab, index=g.index))
    return pd.concat(labels).sort_index()

wr["label"] = label_year_percentiles(wr, year_col="merge_year", diff_col="ppg_diff", low_q=0.2, high_q=0.8)

# Drop rows without labels or year
wr = wr.dropna(subset=["label", "merge_year"]).copy()
wr["merge_year"] = pd.to_numeric(wr["merge_year"], errors="coerce")
wr = wr.dropna(subset=["merge_year"])

# =========================
# Features (same style as TE; keep only existing numeric)
# =========================
feature_candidates = [
    "age","g","gs","tgt","rec","yds","td","x1d","ybc","ybc_r","yac","yac_r","adot",
    "brk_tkl","rec_br","drop","drop_percent","int","rat",
    "avg_cushion","avg_separation","avg_intended_air_yards","percent_share_of_intended_air_yards",
    # naming variants you mentioned:
    "receptions_x","targets_x","avg_yac","avg_expected_yac","avg_yac_above_expectation",
    "receiving_fumbles","receiving_fumbles_lost","receiving_air_yards","receiving_epa","receiving_2pt_conversions",
    "racr","target_share","air_yards_share","wopr_x","fantasy_points",
    "games","tgt_sh","ay_sh","yac_sh","wopr_y","ry_sh","rtd_sh","rfd_sh","dom","w8dom","yptmpa","ppr_sh",
    "height","weight","draft_round","draft_pick",
    "offense_snaps","team_snaps","AVG","adp_percentile",
    "expected_ppr_pg_curr_hist","Games_G"
]

# Keep only columns that exist and are numeric
features = [c for c in feature_candidates if c in wr.columns and pd.api.types.is_numeric_dtype(wr[c])]
if not features:
    raise ValueError("No numeric feature columns found from the candidate list.")

X_all = wr[features]
y_all = wr["label"].astype("category")

# =========================
# Year-based split (train/val/test)
# =========================
train_mask = wr["merge_year"].between(2016, 2020)
val_mask   = wr["merge_year"].between(2021, 2023)
test_mask  = wr["merge_year"].eq(2024)

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_val,   y_val   = X_all[val_mask],   y_all[val_mask]
X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

print("Class counts (train):")
print(y_train.value_counts())
print("\nClass counts (val):")
print(y_val.value_counts())
print("\nClass counts (test):")
print(y_test.value_counts())

# =========================
# Impute
# =========================
imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_val_i   = imp.transform(X_val)
X_test_i  = imp.transform(X_test)

# =========================
# Random Forest (balanced)
# =========================
rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1
)
rf.fit(X_train_i, y_train)

# =========================
# Evaluation
# =========================
def show_results(split, y_true, y_pred):
    print(f"\n{split} Performance:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

y_val_pred = rf.predict(X_val_i)
y_test_pred = rf.predict(X_test_i)

show_results("Validation", y_val, y_val_pred)
show_results("Test", y_test, y_test_pred)

# =========================
# Probabilities & Top Lists (Test 2024)
# =========================
if len(X_test) > 0:
    proba = rf.predict_proba(X_test_i)
    # Map probabilities to class names in the same order as rf.classes_
    prob_cols = [f"prob_{cls}" for cls in rf.classes_]
    prob_df = pd.DataFrame(proba, columns=prob_cols, index=X_test.index)

    prob_df["Player_fixed"] = wr.loc[X_test.index, "Player_fixed"].values
    prob_df["display_name"] = wr.loc[X_test.index, "display_name"].values
    prob_df["FantPos"]      = wr.loc[X_test.index, "FantPos"].values
    prob_df["merge_year"]   = wr.loc[X_test.index, "merge_year"].values

    # Safety: if classes_ order is not ['neutral','over','under'], align helpers dynamically
    cls_to_col = {cls: f"prob_{cls}" for cls in rf.classes_}

    # Top over
    prob_df_over = prob_df.sort_values(cls_to_col.get("over", prob_cols[0]), ascending=False).head(15)
    print("\nTop 15 WRs by probability of OVER-performing (Test 2024):")
    cols_to_show = ["Player_fixed","display_name","FantPos","merge_year",
                    cls_to_col.get("neutral", prob_cols[0]),
                    cls_to_col.get("over", prob_cols[0]),
                    cls_to_col.get("under", prob_cols[0])]
    print(prob_df_over[cols_to_show])

    # Top under
    prob_df_under = prob_df.sort_values(cls_to_col.get("under", prob_cols[-1]), ascending=False).head(15)
    print("\nTop 15 WRs by probability of UNDER-performing (Test 2024):")
    print(prob_df_under[cols_to_show])
else:
    print("\n(No 2024 test rows found after filtering.)")
