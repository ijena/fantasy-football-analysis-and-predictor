import pandas as pd 
import joblib

#loading all data for the models
qb_model_2025_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_qb_vet_data.csv")
rb_model_2025_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_rb_vet_data.csv")
te_model_2025_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_te_vet_data.csv")
wr_model_2025_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_wr_vet_data.csv")

#filtering out data for just 2025

qb_model_2025_df=qb_model_2025_df[qb_model_2025_df["merge_year"]==2025]
rb_model_2025_df=rb_model_2025_df[rb_model_2025_df["merge_year"]==2025]
te_model_2025_df=te_model_2025_df[te_model_2025_df["merge_year"]==2025]
wr_model_2025_df=wr_model_2025_df[wr_model_2025_df["merge_year"]==2025]

#load all QB models

qb_rf_classification_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\qb_model_random_forest_classification.pkl")
qb_rf_classification_percentile_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\qb_model_random_forest_classification_percentile.pkl")
qb_xgb_classification_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\qb_model_xg_boost_classification.pkl")
qb_xgb_classification_percentile_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\qb_model_xg_boost_classification_percentile.pkl")

#load all RB models

rb_rf_classification_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\rb_model_random_forest_classification.pkl")
rb_rf_classification_percentile_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\rb_model_random_forest_classification_percentile.pkl")
rb_xgb_classification_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\rb_model_xg_boost_classification.pkl")
rb_xgb_classification_percentile_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\rb_model_xg_boost_classification_percentile.pkl")


#load all TE models
te_rf_classification_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\te_model_random_forest_classification.pkl")
te_rf_classification_percentile_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\te_model_random_forest_classification_percentile.pkl")
te_xgb_classification_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\te_model_xg_boost_classification.pkl")
te_xgb_classification_percentile_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\te_model_xg_boost_classification_percentile.pkl")

#load all WR models
wr_rf_classification_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\wr_model_random_forest_classification.pkl")
wr_rf_classification_percentile_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\wr_model_random_forest_classification_percentile.pkl")
wr_xgb_classification_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\wr_model_xg_boost_classification.pkl")
wr_xgb_classification_percentile_model = joblib.load(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\models\wr_model_xg_boost_classification_percentile.pkl")

#load all the features of the models

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

rb_features = ["att","yds_x","td_x","x1d_x","ybc_x","ybc_att","yac_x","yac_att","brk_tkl_x",
           "att_br","efficiency","percent_attempts_gte_eight_defenders",
           "expected_rush_yards","rush_yards_over_expected",
           "rush_yards_over_expected_per_att","rush_pct_over_expected","rushing_fumbles",
           "rushing_fumbles_lost","rushing_epa","rushing_2pt_conversions","fantasy_points_x",
           "fantasy_points_ppr_x","games_x","height_x","weight_x","draft_round_x",	
           "draft_pick_x","tgt","yds_y","td_y","x1d_y","ybc_y","ybc_r","yac_y","yac_r",
           "adot","brk_tkl_y","rec_br","drop","drop_percent","int","rat","receiving_fumbles",
           "receiving_fumbles_lost","receiving_air_yards","receiving_epa","receiving_2pt_conversions",
           "racr","target_share","air_yards_share","wopr_x","tgt_sh","ay_sh","yac_sh",	
           "wopr_y","ry_sh","rtd_sh","rfd_sh","dom","w8dom","yptmpa","ppr_sh","offense_snaps",
           "team_snaps","AVG","adp_percentile","adp_percentile_pos","expected_ppr_pg_curr_hist",
           "Games_G"]

te_features = ["age","g","gs","tgt","rec","yds","td","x1d","ybc","ybc_r","yac","yac_r","adot",
            "brk_tkl","rec_br","drop","drop_percent","int","rat","avg_cushion","avg_separation",
            "avg_intended_air_yards","percent_share_of_intended_air_yards","catch_percentage",
            "avg_expected_yac","avg_yac_above_expectation","receiving_fumbles",
            "receiving_fumbles_lost","receiving_air_yards","receiving_epa","receiving_2pt_conversions",
            "racr","target_share","air_yards_share","fantasy_points","fantasy_points_ppr",
            "games","tgt_sh","ay_sh","yac_sh","wopr_y","ry_sh","rtd_sh","rfd_sh","dom",
            "w8dom","yptmpa","ppr_sh","height","weight","draft_round","draft_pick",
            "offense_snaps","team_snaps","offense_snap_percentage","AVG","adp_percentile",
            "adp_percentile_pos","expected_ppr_pg_curr_hist","Games_G"]

wr_features = [
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

