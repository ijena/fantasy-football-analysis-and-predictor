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

# ---------------- Feature Sets (fallback) ----------------
def numeric_feats(df, candidates):
    return [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

qb_features = numeric_feats(qb_df, ["rank","qbr_total","pts_added","qb_plays","epa_total","pass","run",
    "exp_sack","penalty","qbr_raw","sack","pass_attempts","throwaways","spikes","drops","drop_pct",
    "bad_throws","bad_throw_pct","pocket_time","times_blitzed","times_hurried","times_hit",
    "times_pressured","pressure_pct","batted_balls","on_tgt_throws","on_tgt_pct","rpo_plays","rpo_yards",
    "rpo_pass_att","rpo_pass_yards","rpo_rush_att","rpo_rush_yards","pa_pass_att","pa_pass_yards",
    "avg_time_to_throw","avg_completed_air_yards","avg_intended_air_yards","avg_air_yards_differential",
    "aggressiveness","max_completed_air_distance","avg_air_yards_to_sticks","pass_yards","pass_touchdowns",
    "interceptions_x","passer_rating","completion_percentage","expected_completion_percentage",
    "completion_percentage_above_expectation","avg_air_distance","max_air_distance","sacks","sack_yards",
    "sack_fumbles","sack_fumbles_lost","passing_air_yards","passing_yards_after_catch","passing_epa",
    "passing_2pt_conversions","pacr","dakota","carries","rushing_yards","rushing_tds","rushing_fumbles_x",
    "rushing_fumbles_lost_x","rushing_first_downs","rushing_epa_x","rushing_2pt_conversions_x",
    "fantasy_points_x","games_x","ppr_sh","height_x","weight_x","draft_round_x","draft_pick_x","age","gs",
    "ybc","ybc_att","yac","yac_att","brk_tkl","att_br","offense_snaps","team_snaps","AVG","adp_percentile",
    "adp_percentile_pos","expected_ppr_pg_curr_hist","Games_G"])

rb_features = numeric_feats(rb_df, [
    "att","yds_x","td_x","x1d_x","ybc_x","ybc_att","yac_x","yac_att","brk_tkl_x","att_br","efficiency",
    "percent_attempts_gte_eight_defenders","expected_rush_yards","rush_yards_over_expected",
    "rush_yards_over_expected_per_att","rush_pct_over_expected","rushing_fumbles","rushing_fumbles_lost",
    "rushing_epa","rushing_2pt_conversions","fantasy_points_x","fantasy_points_ppr_x","games_x","height_x",
    "weight_x","draft_round_x","draft_pick_x","tgt","yds_y","td_y","x1d_y","ybc_y","ybc_r","yac_y","yac_r",
    "adot","brk_tkl_y","rec_br","drop","drop_percent","int","rat","receiving_fumbles","receiving_fumbles_lost",
    "receiving_air_yards","receiving_epa","receiving_2pt_conversions","racr","target_share","air_yards_share",
    "wopr_x","tgt_sh","ay_sh","yac_sh","wopr_y","ry_sh","rtd_sh","rfd_sh","dom","w8dom","yptmpa","ppr_sh",
    "offense_snaps","team_snaps","AVG","adp_percentile","adp_percentile_pos","expected_ppr_pg_curr_hist","Games_G"])

te_features = numeric_feats(te_df, ["age","g","gs","tgt","rec","yds","td","x1d","ybc","ybc_r","yac","yac_r","adot",
    "brk_tkl","rec_br","drop","drop_percent","int","rat","avg_cushion","avg_separation","avg_intended_air_yards",
    "percent_share_of_intended_air_yards","catch_percentage","avg_expected_yac","avg_yac_above_expectation",
    "receiving_fumbles","receiving_fumbles_lost","receiving_air_yards","receiving_epa","receiving_2pt_conversions",
    "racr","target_share","air_yards_share","fantasy_points","fantasy_points_ppr","games","tgt_sh","ay_sh","yac_sh",
    "wopr_y","ry_sh","rtd_sh","rfd_sh","dom","w8dom","yptmpa","ppr_sh","height","weight","draft_round","draft_pick",
    "offense_snaps","team_snaps","AVG","adp_percentile","adp_percentile_pos","expected_ppr_pg_curr_hist","Games_G"])

wr_features = numeric_feats(wr_df, ["age","g","gs","tgt","rec","yds","td","x1d","ybc","ybc_r","yac","yac_r","adot",
    "brk_tkl","rec_br","drop","drop_percent","int","rat","avg_cushion","avg_separation","avg_intended_air_yards",
    "percent_share_of_intended_air_yards","receptions_x","targets_x","avg_yac","avg_expected_yac",
    "avg_yac_above_expectation","receiving_fumbles","receiving_fumbles_lost","receiving_air_yards","receiving_epa",
    "receiving_2pt_conversions","racr","target_share","air_yards_share","wopr_x","fantasy_points","games","tgt_sh",
    "ay_sh","yac_sh","wopr_y","ry_sh","rtd_sh","rfd_sh","dom","w8dom","yptmpa","ppr_sh","height","weight",
    "draft_round","draft_pick","offense_snaps","team_snaps","AVG","adp_percentile","expected_ppr_pg_curr_hist","Games_G"])

# ---------------- Models to Run ----------------
MODELS = {
    "QB": {"data": qb_df, "features": qb_features, "models": {
        "xgb_classification": os.path.join(MODEL_DIR, "qb_model_xg_boost_classification.pkl"),
        "xgb_percentile":     os.path.join(MODEL_DIR, "qb_model_xg_boost_classification_percentile.pkl"),
        "rf_classification":  os.path.join(MODEL_DIR, "qb_model_random_forest_classification.pkl"),
        "rf_percentile":      os.path.join(MODEL_DIR, "qb_model_random_forest_classification_percentile.pkl"),
    }},
    "RB": {"data": rb_df, "features": rb_features, "models": {
        "xgb_classification": os.path.join(MODEL_DIR, "rb_model_xg_boost_classification.pkl"),
        "xgb_percentile":     os.path.join(MODEL_DIR, "rb_model_xg_boost_classification_percentile.pkl"),
        "rf_classification":  os.path.join(MODEL_DIR, "rb_model_random_forest_classification.pkl"),
        "rf_percentile":      os.path.join(MODEL_DIR, "rb_model_random_forest_classification_percentile.pkl"),
    }},
    "TE": {"data": te_df, "features": te_features, "models": {
        "xgb_classification": os.path.join(MODEL_DIR, "te_model_xg_boost_classification.pkl"),
        "xgb_percentile":     os.path.join(MODEL_DIR, "te_model_xg_boost_classification_percentile.pkl"),
        "rf_classification":  os.path.join(MODEL_DIR, "te_model_random_forest_classification.pkl"),
        "rf_percentile":      os.path.join(MODEL_DIR, "te_model_random_forest_classification_percentile.pkl"),
    }},
    "WR": {"data": wr_df, "features": wr_features, "models": {
        "xgb_classification": os.path.join(MODEL_DIR, "wr_model_xg_boost_classification.pkl"),
        "xgb_percentile":     os.path.join(MODEL_DIR, "wr_model_xg_boost_classification_percentile.pkl"),
        "rf_classification":  os.path.join(MODEL_DIR, "wr_model_random_forest_classification.pkl"),
        "rf_percentile":      os.path.join(MODEL_DIR, "wr_model_random_forest_classification_percentile.pkl"),
    }},
}

DEFAULT_CLASS_ORDER = ["neutral", "over", "under"]

# ---------------- Helpers ----------------
def detect_team_col(df):
    for cand in ["team","Team","recent_team","team_abbr","recent_team_abbr","Tm"]:
        if cand in df.columns:
            return cand
    return None

def get_model_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "get_booster"):
        try:
            fn = model.get_booster().feature_names
            if fn: return list(fn)
        except: pass
    return None

def align_X(df, fallback_feats, model):
    feat_names = get_model_feature_names(model) or fallback_feats
    for m in [f for f in feat_names if f not in df.columns]:
        df[m] = 0
    return df[feat_names].fillna(0), feat_names

def class_names_from_model(model):
    classes_ = getattr(model, "classes_", None)
    if classes_ is None: return DEFAULT_CLASS_ORDER
    return [c.decode() if isinstance(c, bytes) else c for c in classes_]

def run_models_for_position(pos_name, cfg):
    df_pos = cfg["data"].copy()
    feats  = cfg["features"]
    if df_pos.empty:
        print(f"\n==== {pos_name}: No rows for 2025. Skipping. ====")
        return pd.DataFrame()

    team_col = detect_team_col(df_pos)
    id_df = pd.DataFrame({
        "Player_fixed": df_pos["Player_fixed"],
        "merge_year":   df_pos["merge_year"],
        "position":     pos_name
    })
    id_df["team_name"] = df_pos[team_col] if team_col else ""

    results = id_df.copy()

    for model_key, model_path in cfg["models"].items():
        print(f"\n[{pos_name}] Loading model: {model_key}")
        model = joblib.load(model_path)
        X, _ = align_X(df_pos.copy(), feats, model)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            cls_names = class_names_from_model(model)
            if probs.shape[1] != len(cls_names):
                cls_names = [f"class_{i}" for i in range(probs.shape[1])]
            for i, cls in enumerate(cls_names):
                colname = f"{model_key}_prob_{cls}"
                results[colname] = probs[:, i]
        else:
            results[f"{model_key}_label"] = model.predict(X)

    out_path = os.path.join(OUT_DIR, f"{pos_name}_MODEL_PROBS_2025.csv")
    results.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")
    return results

# ---------------- Run everything ----------------
qb_res = run_models_for_position("QB", MODELS["QB"])
rb_res = run_models_for_position("RB", MODELS["RB"])
te_res = run_models_for_position("TE", MODELS["TE"])
wr_res = run_models_for_position("WR", MODELS["WR"])

grand = pd.concat([x for x in [qb_res, rb_res, te_res, wr_res] if not x.empty], ignore_index=True)
grand_path = os.path.join(OUT_DIR, "ALL_POSITIONS_MODEL_PROBS_2025.csv")
grand.to_csv(grand_path, index=False)
print(f"\n🎉 Saved grand combined: {grand_path}")
