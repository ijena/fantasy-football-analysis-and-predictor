import os
import pandas as pd
import numpy as np
import joblib

# ---------------- Paths ----------------
DATA_DIR   = r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data"
MODEL_DIR  = r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models"
OUT_DIR    = r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\predictions_2025"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Load 2025 Data ----------------
qb_df = pd.read_csv(os.path.join(DATA_DIR, "master_qb_vet_data.csv"))
rb_df = pd.read_csv(os.path.join(DATA_DIR, "master_rb_vet_data.csv"))
te_df = pd.read_csv(os.path.join(DATA_DIR, "master_te_vet_data.csv"))
wr_df = pd.read_csv(os.path.join(DATA_DIR, "master_wr_vet_data.csv"))

qb_df = qb_df[qb_df["merge_year"] == 2025].reset_index(drop=True)
rb_df = rb_df[rb_df["merge_year"] == 2025].reset_index(drop=True)
te_df = te_df[te_df["merge_year"] == 2025].reset_index(drop=True)
wr_df = wr_df[wr_df["merge_year"] == 2025].reset_index(drop=True)

# ---------------- Feature Sets ----------------
qb_features = ["rank","qbr_total","pts_added","qb_plays","epa_total","pass","run",
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
qb_features = [c for c in qb_features if pd.api.types.is_numeric_dtype(qb_df[c])]


rb_features = [
    "att","yds_x","td_x","x1d_x","ybc_x","ybc_att","yac_x","yac_att","brk_tkl_x",
    "att_br","efficiency","percent_attempts_gte_eight_defenders",
    "expected_rush_yards","rush_yards_over_expected","rush_yards_over_expected_per_att",
    "rush_pct_over_expected","rushing_fumbles","rushing_fumbles_lost","rushing_epa",
    "rushing_2pt_conversions","fantasy_points_x","fantasy_points_ppr_x","games_x",
    "height_x","weight_x","draft_round_x","draft_pick_x","tgt","yds_y","td_y","x1d_y",
    "ybc_y","ybc_r","yac_y","yac_r","adot","brk_tkl_y","rec_br","drop","drop_percent","int","rat"
    "receiving_fumbles","receiving_fumbles_lost","receiving_air_yards","receiving_epa",
    "receiving_2pt_conversions","racr","target_share","air_ysards_share","wopr_x","tgt_sh",
    "ay_sh","yac_sh","wopr_y","ry_sh","rtd_sh","rfd_sh","dom","w8dom","yptmpa","ppr_sh",
    "offense_snaps","team_snaps","AVG","adp_percentile","adp_percentile_pos",
    "expected_ppr_pg_curr_hist","Games_G"
]

rb_features = [c for c in rb_features if pd.api.types.is_numeric_dtype(rb_df[c])]


te_features = ["age","g","gs","tgt","rec","yds","td","x1d","ybc","ybc_r","yac","yac_r","adot",
    "brk_tkl","rec_br","drop","drop_percent","int","rat","avg_cushion","avg_separation",
    "avg_intended_air_yards","percent_share_of_intended_air_yards","catch_percentage",
    "avg_expected_yac","avg_yac_above_expectation","receiving_fumbles",
    "receiving_fumbles_lost","receiving_air_yards","receiving_epa","receiving_2pt_conversions",
    "racr","target_share","air_yards_share","fantasy_points","fantasy_points_ppr",
    "games","tgt_sh","ay_sh","yac_sh","wopr_y","ry_sh","rtd_sh","rfd_sh","dom","w8dom","yptmpa",
    "ppr_sh","height","weight","draft_round","draft_pick","offense_snaps","team_snaps",
    "AVG","adp_percentile","adp_percentile_pos",
    "expected_ppr_pg_curr_hist","Games_G"]
te_features = [c for c in te_features if pd.api.types.is_numeric_dtype(te_df[c])]


wr_features = ["age","g","gs","tgt","rec","yds","td","x1d","ybc","ybc_r","yac","yac_r","adot",
    "brk_tkl","rec_br","drop","drop_percent","int","rat","avg_cushion","avg_separation",
    "avg_intended_air_yards","percent_share_of_intended_air_yards","receptions_x","targets_x",
    "avg_yac","avg_expected_yac","avg_yac_above_expectation","receiving_fumbles",
    "receiving_fumbles_lost","receiving_air_yards","receiving_epa","receiving_2pt_conversions",
    "racr","target_share","air_yards_share","wopr_x","fantasy_points","games","tgt_sh","ay_sh",
    "yac_sh","wopr_y","ry_sh","rtd_sh","rfd_sh","dom","w8dom","yptmpa","ppr_sh","height","weight",
    "draft_round","draft_pick","offense_snaps","team_snaps","AVG","adp_percentile",
    "expected_ppr_pg_curr_hist","Games_G"]
wr_features = [c for c in wr_features if pd.api.types.is_numeric_dtype(wr_df[c])]


# ---------------- Models to Run (per position) ----------------
MODELS = {
    "QB": {
        "data": qb_df,
        "features": qb_features,
        "models": {
            "xgb_classification":      os.path.join(MODEL_DIR, "qb_model_xg_boost_classification.pkl"),
            "xgb_percentile":          os.path.join(MODEL_DIR, "qb_model_xg_boost_classification_percentile.pkl"),
            "rf_classification":       os.path.join(MODEL_DIR, "qb_model_random_forest_classification.pkl"),
            "rf_percentile":           os.path.join(MODEL_DIR, "qb_model_random_forest_classification_percentile.pkl"),
        },
        # optional per-position encoders (if you saved them separately)
        "encoder": os.path.join(MODEL_DIR, "qb_label_encoder.pkl"),
    },
    "RB": {
        "data": rb_df,
        "features": rb_features,
        "models": {
            "xgb_classification":      os.path.join(MODEL_DIR, "rb_model_xg_boost_classification.pkl"),
            "xgb_percentile":          os.path.join(MODEL_DIR, "rb_model_xg_boost_classification_percentile.pkl"),
            "rf_classification":       os.path.join(MODEL_DIR, "rb_model_random_forest_classification.pkl"),
            "rf_percentile":           os.path.join(MODEL_DIR, "rb_model_random_forest_classification_percentile.pkl"),
        },
        "encoder": os.path.join(MODEL_DIR, "rb_label_encoder.pkl"),
    },
    "TE": {
        "data": te_df,
        "features": te_features,
        "models": {
            "xgb_classification":      os.path.join(MODEL_DIR, "te_model_xg_boost_classification.pkl"),
            "xgb_percentile":          os.path.join(MODEL_DIR, "te_model_xg_boost_classification_percentile.pkl"),
            "rf_classification":       os.path.join(MODEL_DIR, "te_model_random_forest_classification.pkl"),
            "rf_percentile":           os.path.join(MODEL_DIR, "te_model_random_forest_classification_percentile.pkl"),
        },
        "encoder": os.path.join(MODEL_DIR, "te_label_encoder.pkl"),
    },
    "WR": {
        "data": wr_df,
        "features": wr_features,
        "models": {
            "xgb_classification":      os.path.join(MODEL_DIR, "wr_model_xg_boost_classification.pkl"),
            "xgb_percentile":          os.path.join(MODEL_DIR, "wr_model_xg_boost_classification_percentile.pkl"),
            "rf_classification":       os.path.join(MODEL_DIR, "wr_model_random_forest_classification.pkl"),
            "rf_percentile":           os.path.join(MODEL_DIR, "wr_model_random_forest_classification_percentile.pkl"),
        },
        "encoder": os.path.join(MODEL_DIR, "wr_label_encoder.pkl"),
    },
}

# Fallback class order if we can't load an encoder or the model holds strings already
DEFAULT_CLASS_ORDER = ["neutral", "over", "under"]

def align_X(df, features):
    # ensure all features exist; add missing with 0
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"⚠️  Missing features filled with 0: {missing}")
        for m in missing:
            df[m] = 0
    return df[features].fillna(0)

def get_class_names(model, encoder_path):
    """
    Returns a list of class names in the same order as model.predict_proba columns.
    Priority:
      1) If model.classes_ are strings, just return them
      2) If we can load a LabelEncoder, map indices -> original labels via encoder.classes_
      3) Fallback to DEFAULT_CLASS_ORDER (must match your training convention)
    """
    # model.classes_ exists for sklearn/xgb; could be [0,1,2] or strings
    model_classes = getattr(model, "classes_", None)

    # if classes_ are already strings
    if model_classes is not None and all(isinstance(c, str) for c in model_classes):
        return list(model_classes)

    # try to load a label encoder
    if os.path.exists(encoder_path):
        try:
            le = joblib.load(encoder_path)
            # model.classes_ give the indices; map back to label strings
            if model_classes is None:
                # some models may not expose classes_; assume [0..n-1]
                return list(le.classes_)
            else:
                return [le.classes_[i] for i in model_classes]
        except Exception:
            pass

    # last resort: default order
    return DEFAULT_CLASS_ORDER

def run_models_for_position(pos_name, cfg):
    df_pos = cfg["data"].copy()
    feats  = cfg["features"]
    enc_path = cfg.get("encoder", "")

    if df_pos.empty:
        print(f"\n==== {pos_name}: No rows for 2025. Skipping. ====")
        return pd.DataFrame()

    all_results = []

    for model_name, model_path in cfg["models"].items():
        print(f"\n[{pos_name}] Loading model: {model_name}")
        model = joblib.load(model_path)

        class_names = get_class_names(model, enc_path)

        X = align_X(df_pos.copy(), feats)

        # predictions
        y_pred = model.predict(X)

        # try to convert to labels if encoder/classes were numeric
        if isinstance(y_pred[0], (np.integer, int)) and class_names and not isinstance(class_names[0], (np.integer, int)):
            # map numeric -> class label positionally
            pred_labels = [class_names[i] if i < len(class_names) else str(i) for i in y_pred]
        else:
            # already labels (strings)
            pred_labels = y_pred

        # probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            # shape check
            if probs.ndim == 2 and probs.shape[1] == len(class_names):
                prob_cols = [f"prob_{c}" for c in class_names]
            else:
                # mismatch fallback: name generically
                prob_cols = [f"prob_class_{i}" for i in range(probs.shape[1])]
            prob_df = pd.DataFrame(probs, columns=prob_cols)
        else:
            prob_df = pd.DataFrame()

        out = pd.concat([df_pos.reset_index(drop=True), prob_df], axis=1)
        out["pred_label"] = pred_labels
        out["model_name"] = f"{pos_name.lower()}_{model_name}"

        # Save per-model CSV
        out_path = os.path.join(OUT_DIR, f"{pos_name.lower()}_{model_name}_2025.csv")
        out.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path}")

        all_results.append(out)

    # Combined per-position CSV
    if all_results:
        combo = pd.concat(all_results, axis=0, ignore_index=True)
        combo_path = os.path.join(OUT_DIR, f"{pos_name.lower()}_ALLMODELS_2025.csv")
        combo.to_csv(combo_path, index=False)
        print(f"✅ Saved combined: {combo_path}")
        return combo
    return pd.DataFrame()

# ---------------- Run everything ----------------
qb_all  = run_models_for_position("QB", MODELS["QB"])
rb_all  = run_models_for_position("RB", MODELS["RB"])
te_all  = run_models_for_position("TE", MODELS["TE"])
wr_all  = run_models_for_position("WR", MODELS["WR"])

# Optionally, save a grand combined file across all positions & models
grand = pd.concat([x for x in [qb_all, rb_all, te_all, wr_all] if not x.empty], ignore_index=True)
grand_path = os.path.join(OUT_DIR, "ALL_POSITIONS_ALL_MODELS_2025.csv")
grand.to_csv(grand_path, index=False)
print(f"\n🎉 Saved grand combined: {grand_path}")

# Quick sanity prints
for pos_name, df_pos in [("QB", qb_df), ("RB", rb_df), ("TE", te_df), ("WR", wr_df)]:
    print(f"{pos_name} 2025 rows: {len(df_pos)}")
