import pandas as pd

qb_model_data = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_qb_vet_data.csv")


#preparing the dataset in terms of features for training

meta_data = ["season","team_abb","player_id_x","name_short","name_first","name_last",
             "name_display","headshot_href","team_x","qualified","pfr_id_x_x",
             "player_gsis_id_x","player_id_y","display_name","common_first_name",
             "short_name","football_name_x","suffix_x",	"esb_id_x",	"nfl_id_x",
             "pfr_id_y_x","pff_id_x","otc_id_x","smart_id_x","birth_date_x","headshot_x",
             "player_x","pfr_id_x_y","tm","merge_year","Player_fixed"]

features = ["rank","qbr_total","pts_added","qb_plays","epa_total","pass","run",
            "exp_sack",	"penalty",	"qbr_raw","sack","pass_attempts",
            "throwaways","spikes","drops","drop_pct","bad_throws","bad_throw_pct",
            "pocket_time","times_blitzed","times_hurried","times_hit","times_pressured",
            "pressure_pct",	"batted_balls",	"on_tgt_throws","on_tgt_pct","rpo_plays",
            "rpo_yards","rpo_pass_att","rpo_pass_yards","rpo_rush_att","rpo_rush_yards",
            "pa_pass_att","pa_pass_yards","avg_time_to_throw","avg_completed_air_yards",
            "avg_intended_air_yards","avg_air_yards_differential","aggressiveness",
            "max_completed_air_distance","avg_air_yards_to_sticks",	"pass_yards",	
            "pass_touchdowns","interceptions_x","passer_rating","completion_percentage",
            "expected_completion_percentage","completion_percentage_above_expectation",
            "avg_air_distance",	"max_air_distance","sacks",	"sack_yards","sack_fumbles",
            "sack_fumbles_lost","passing_air_yards","passing_yards_after_catch",
            "passing_epa",	"passing_2pt_conversions",	"pacr",	"dakota","carries",	
            "rushing_yards","rushing_tds","rushing_fumbles_x","rushing_fumbles_lost_x",
            "rushing_first_downs","rushing_epa_x","rushing_2pt_conversions_x","fantasy_points_x",
            "games_x","ppr_sh","height_x","weight_x","college_name_x","college_conference_x",	
            "draft_round_x","draft_pick_x","age","gs","ybc","ybc_att","yac","yac_att",	
            "brk_tkl","att_br","offense_snaps","team_snaps","AVG","adp_percentile",
            "adp_percentile_pos","expected_ppr_pg_curr_hist","Games_G"]