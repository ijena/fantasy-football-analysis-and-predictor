import nfl_data_py
import pandas as pd

def load_depth_chart_data(years, fantasy_positions):


    depth_charts = nfl_data_py.import_depth_charts(years)

    #filter for fantasy relevant positions
    #drop irrelevant columns
    depth_charts = depth_charts.drop(columns = ["jersey_number"])
    depth_charts = depth_charts[depth_charts["position"].isin(fantasy_positions)]
    #filter for regular season games only
    depth_charts = depth_charts[depth_charts["game_type"]=="REG"]
    #filter for offense
    depth_charts = depth_charts[depth_charts["formation"]=="Offense"]

    depth_charts.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\depth_chart.csv")   
    return depth_charts
def load_seasonal_data(years):
    season_stats = nfl_data_py.import_seasonal_data(years)
    #drop special teams stats
    season_stats = season_stats.drop(columns = ["special_teams_tds"])
    season_stats.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\season_stats.csv")   
    return season_stats
def load_combine_data(years, fantasy_positions):
    #fantasy_positiions helps us get data only for fantasy relevant positions
    combine_data = nfl_data_py.import_combine_data(years, fantasy_positions)
    #height column is messed up in the dataset and populated with birth data
    combine_data = combine_data.drop(columns = ["ht"])
    #remove undrafted players
    combine_data = combine_data.dropna(subset=["draft_team"])
    combine_data.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\combine_data.csv")   
    return combine_data

def load_draft_picks(years,fantasy_positions):
    draft_pick_data = nfl_data_py.import_draft_picks(years)
    #filter for fantasy relevant players
    draft_pick_data = draft_pick_data[draft_pick_data["side"]=="O"]
    draft_pick_data = draft_pick_data[draft_pick_data["position"].isin(fantasy_positions)]
    #filter for offensive stats only
    draft_pick_data = draft_pick_data.drop(columns=["def_solo_tackles", "def_ints",	"def_sacks"])
    draft_pick_data.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\draft_pick_data.csv")   
    return draft_pick_data
    
def load_ngs_data (stat_type, years):
    ngs_data = nfl_data_py.import_ngs_data(stat_type=stat_type, years=years)
    #filter for just regular season games
    ngs_data = ngs_data[ngs_data["season_type"]=="REG"]
    ngs_data = ngs_data[ngs_data["week"]==0]
    #remove irrelevant columns like jersey number
    ngs_data = ngs_data.drop(columns="player_jersey_number")
    ngs_data.to_csv(fr"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\ngs_{stat_type}_data.csv")
    return ngs_data

def load_player_data(years,fantasy_positions):
    player_data = nfl_data_py.import_players()
    #filter data for player's last season being after 2014
    player_data = player_data[player_data["last_season"].between(years[0],years[len(years)-1]+1)]
    #filter data for fantasy relevant positions"
    player_data = player_data[player_data["position_group"].isin(fantasy_positions)]
    player_data = player_data.drop(columns="jersey_number")
    player_data.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\player_data.csv")
    return player_data
def load_qbr_data(years, level):
    qbr_data = nfl_data_py.import_qbr(years, level)
    #filter out data for just regular season 
    if(level=="nfl"):
        qbr_data = qbr_data[qbr_data["season_type"] == "Regular"]
    qbr_data.to_csv(fr"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\{level}_qbr_data.csv")
    return qbr_data
def load_seasonal_pfr(stats_type):
    #data only available from 2018
    seasonal_pfr_data = nfl_data_py.import_seasonal_pfr(stats_type)
    seasonal_pfr_data.to_csv(fr"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\{stats_type}_seasonal_pfr_data.csv")
    return seasonal_pfr_data
def load_snap_counts(years,fantasy_positions):
    snap_count_data = nfl_data_py.import_snap_counts(years)
    #drop columns for defense and special team stats
    snap_count_data = snap_count_data.drop(columns=["defense_snaps","defense_pct","st_snaps","st_pct",".progress"])
    #filter data only for relevant fantasy positions
    snap_count_data = snap_count_data[snap_count_data["position"].isin(fantasy_positions)]
    #filter for just regular season games
    snap_count_data = snap_count_data[snap_count_data["game_type"]=="REG"]
    
    #calculate season snaps played for each player
    season_agg = (
    snap_count_data.groupby(["team","season", "player","position"], as_index=False)
    .agg({
        "offense_snaps": "sum",
        })
)

    #calculate weekly team snaps
    season_agg_temp= (snap_count_data.groupby(["week","season","team"], as_index = False).agg({"offense_snaps": "max"}))
    season_agg_temp["team_snaps"] = season_agg_temp["offense_snaps"]
    #calculate season team snaps
    temp_season_agg = (season_agg_temp.groupby(["season","team"], as_index = False).agg({"team_snaps": "sum"}))
    #merge season team snaps and season player snaps
    season_snap_counts = season_agg.merge(temp_season_agg, on=["season","team"])
    #calculate percentage of team snaps that a player was involved in
    season_snap_counts["offense_snap_percentage"] = season_snap_counts["offense_snaps"]/season_snap_counts["team_snaps"]
    season_snap_counts.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\season_snap_count_data.csv")
    return season_snap_counts

def load_play_by_play_data(years):
    play_by_play_data = nfl_data_py.import_pbp_data(years)
    #drop irrelevant columns
    play_by_play_data = play_by_play_data.drop(columns=["qb_kneel","qb_spike",
    "pass_length","pass_location","field_goal_result","kick_distance",
    "extra_point_result","two_point_conv_result","home_timeouts_remaining",
    "away_timeouts_remaining","timeout","timeout_team","run_location","run_gap",
    "posteam_timeouts_remaining","defteam_timeouts_remaining","total_home_score",
    "total_away_score",	"posteam_score","defteam_score","score_differential",
    "posteam_score_post","defteam_score_post","score_differential_post","no_score_prob",
    "opp_fg_prob","opp_safety_prob","opp_td_prob","fg_prob","safety_prob","td_prob",
    "extra_point_prob","two_point_conversion_prob","total_home_epa","total_away_epa",
    "total_home_rush_epa","total_away_rush_epa","total_home_pass_epa","total_away_pass_epa",
    "wp","def_wp","home_wp","away_wp","wpa","vegas_wpa","vegas_home_wpa","home_wp_post",
    "away_wp_post","vegas_wp","vegas_home_wp","total_home_rush_wpa","total_away_rush_wpa",
    "total_home_pass_wpa","total_away_pass_wpa","air_wpa","yac_wpa","comp_air_wpa",	
    "comp_yac_wpa","total_home_comp_air_wpa","total_away_comp_air_wpa",
    "total_home_comp_yac_wpa","total_away_comp_yac_wpa","total_home_raw_air_wpa",	
    "total_away_raw_air_wpa","total_home_raw_yac_wpa","total_away_raw_yac_wpa",
    "first_down_penalty","incomplete_pass",	"touchback","interception",
    "punt_inside_twenty","punt_in_endzone","punt_out_of_bounds","punt_downed",	
    "punt_fair_catch","kickoff_inside_twenty","kickoff_in_endzone",
    "kickoff_out_of_bounds","kickoff_downed","kickoff_fair_catch", "fumble_forced",
    "fumble_not_forced","fumble_out_of_bounds",	"solo_tackle","safety",	"penalty",
    "tackled_for_loss","own_kickoff_recovery","own_kickoff_recovery_td","qb_hit",
    "fumble_lost","return_touchdown","extra_point_attempt","two_point_attempt",
    "field_goal_attempt","kickoff_attempt",	"punt_attempt","assist_tackle",
    "lateral_reception","lateral_rush",	"lateral_return","lateral_recovery",
    "lateral_receiver_player_id","lateral_receiver_player_name",
    "lateral_receiving_yards","lateral_rusher_player_id","lateral_rusher_player_name",
    "lateral_rushing_yards","lateral_sack_player_id","lateral_sack_player_name",
    "interception_player_id","interception_player_name","lateral_interception_player_id",
    "lateral_interception_player_name","punt_returner_player_id","punt_returner_player_name",
    "lateral_punt_returner_player_id","lateral_punt_returner_player_name",
    "kickoff_returner_player_name","kickoff_returner_player_id","lateral_kickoff_returner_player_id",
    "lateral_kickoff_returner_player_name","punter_player_id","punter_player_name",	
    "kicker_player_name","kicker_player_id","own_kickoff_recovery_player_id",
    "own_kickoff_recovery_player_name",	"blocked_player_id","blocked_player_name",
    "tackle_for_loss_1_player_id","tackle_for_loss_1_player_name","tackle_for_loss_2_player_id",	
    "tackle_for_loss_2_player_name","qb_hit_1_player_id","qb_hit_1_player_name","qb_hit_2_player_id",
    "qb_hit_2_player_name","forced_fumble_player_1_team","forced_fumble_player_1_player_id",
    "forced_fumble_player_1_player_name","forced_fumble_player_2_team",	
    "forced_fumble_player_2_player_id","forced_fumble_player_2_player_name","solo_tackle_1_team",
    "solo_tackle_2_team","solo_tackle_1_player_id",	"solo_tackle_2_player_id","solo_tackle_1_player_name",
    "solo_tackle_2_player_name","assist_tackle_1_player_id","assist_tackle_1_player_name","assist_tackle_1_team",
    "assist_tackle_2_player_id","assist_tackle_2_player_name","assist_tackle_2_team","assist_tackle_3_player_id",
    "assist_tackle_3_player_name","assist_tackle_3_team","assist_tackle_4_player_id","assist_tackle_4_player_name",
    "assist_tackle_4_team","tackle_with_assist","tackle_with_assist_1_player_id","tackle_with_assist_1_player_name",
    "tackle_with_assist_1_team","tackle_with_assist_2_player_id","tackle_with_assist_2_player_name",
    "tackle_with_assist_2_team","pass_defense_1_player_id",	"pass_defense_1_player_name","pass_defense_2_player_id",
    "pass_defense_2_player_name","fumbled_1_team","fumbled_1_player_id","fumbled_1_player_name","fumbled_2_player_id",
    "fumbled_2_player_name","fumbled_2_team","fumble_recovery_1_team","fumble_recovery_1_yards","fumble_recovery_1_player_id",
    "fumble_recovery_1_player_name","fumble_recovery_2_team","fumble_recovery_2_yards",	"fumble_recovery_2_player_id",
    "fumble_recovery_2_player_name","sack_player_id","sack_player_name","half_sack_1_player_id","half_sack_1_player_name",
    "half_sack_2_player_id","half_sack_2_player_name","return_team","return_yards","penalty_team","penalty_player_id",
    "penalty_player_name","penalty_yards","replay_or_challenge","replay_or_challenge_result","penalty_type",
    "defensive_two_point_attempt","defensive_two_point_conv","defensive_extra_point_attempt","defensive_extra_point_conv",
    "safety_player_name","safety_player_id","series","series_success","series_result","order_sequence",	"start_time",
    "time_of_day","stadium","weather","play_clock","play_deleted","play_type_nfl","special_teams_play",	"st_play_type",
    "end_clock_time","end_yard_line","fixed_drive",	"fixed_drive_result","drive_real_start_time", "drive_quarter_start",
    "drive_quarter_end","drive_yards_penalized","drive_start_transition","drive_end_transition","drive_game_clock_start",
    "drive_game_clock_end",	"drive_start_yard_line","drive_end_yard_line","drive_play_id_started","drive_play_id_ended",
    "away_score","home_score","location","result","home_coach","away_coach","stadium_id","game_stadium","aborted_play",
    "success","passer_jersey_number","rusher_jersey_number","receiver_jersey_number","special", "jersey_number",
    "out_of_bounds","home_opening_kickoff",	"possession_team",	"offense_personnel","defense_personnel","number_of_pass_rushers",
    "offense_names","defense_names","offense_positions","defense_positions","offense_numbers","defense_numbers"])
    #filter for regular season game

    play_by_play_data = play_by_play_data[play_by_play_data["season_type"]=="REG"]
    #filter for relevant plays
    play_by_play_data = play_by_play_data[play_by_play_data["play_type"].isin(["run","pass"])]
        
    play_by_play_data.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\play_by_play_data.csv")
    return play_by_play_data

def merge_season_stats_player_data(season_stats, player_data):
    merged_season_stats_player_data = season_stats.merge(player_data,left_on =["player_id"], right_on = ["gsis_id"],
                                                         how="inner")
    merged_season_stats_player_data["rookie"] = (merged_season_stats_player_data["rookie_season"]==merged_season_stats_player_data["season"]).astype(int)
    merged_season_stats_player_data.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\merged_season_stats_player_data.csv")
    return merged_season_stats_player_data

def merge_passing_data(merged_season_stats_player_data,nfl_qbr_data,seasonal_pfr_pass_data,ngs_data_passing):
    #merging advanced qbr stats with their counting stats for each qb and season
    merged_nfl_qbr_data_pass_seasonal_pfr_data = nfl_qbr_data.merge(seasonal_pfr_pass_data,left_on=["name_display","season"],
                                                                  right_on=["player","season"],how="right")

    
    #merging next gen stats for qbs with the above dataset
    merged_nfl_qbr_data_pass_seasonal_pfr_ngs_passing_data = merged_nfl_qbr_data_pass_seasonal_pfr_data.merge(ngs_data_passing,left_on=["name_display","season"],right_on=["player_display_name","season"],how="outer")

    #filtering out just the qb stats from the merged season stats and player data
    merged_qb_season_stats_player_data= merged_season_stats_player_data[merged_season_stats_player_data["position"]=="QB"]
    
    
    #create a master passing dataset with all the passing data except college qb data
    master_passing_data = merged_nfl_qbr_data_pass_seasonal_pfr_ngs_passing_data.merge(merged_qb_season_stats_player_data, left_on=["player_display_name","season"],right_on=["display_name","season"], how ="right")
    master_passing_data= master_passing_data.drop(columns=["season_type_x","game_week","player","team_y","intended_air_yards","intended_air_yards_per_pass_attempt","completed_air_yards","completed_air_yards_per_completion"
                                      ,"completed_air_yards_per_pass_attempt","pass_yards_after_catch",	"pass_yards_after_catch_per_completion","scrambles","scramble_yards_per_attempt","season_type_y",
                                      "week","player_display_name",	"player_position","team_abbr","player_first_name","player_last_name","player_short_name","season_type","receptions",
                                      "targets","receiving_yards","receiving_tds","receiving_fumbles","receiving_fumbles_lost","receiving_air_yards","receiving_yards_after_catch",	"receiving_first_downs",	
                                      "receiving_epa","receiving_2pt_conversions","racr","target_share","air_yards_share","wopr_x","tgt_sh","ay_sh","yac_sh","wopr_y","ry_sh","rtd_sh","rfd_sh","rtdfd_sh",
                                      "dom","w8dom","yptmpa","first_name","last_name","position_group","position","ngs_position_group","ngs_position"])
    master_passing_data.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\master_passing_data.csv")
    return master_passing_data

def merge_rushing_data(seasonal_pfr_rush_data,ngs_rushing_data,merged_season_stats_player_data):
    merged_seasonal_pfr_ngs_rush_data = seasonal_pfr_rush_data.merge(ngs_rushing_data, left_on=["player","season"],right_on= ["player_display_name","season"],how="outer")
    # merged_seasonal_pfr_ngs_rush_data.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\test1.csv")
    master_rushing_data = merged_seasonal_pfr_ngs_rush_data.merge(merged_season_stats_player_data, how="right",left_on =["player","season"],right_on=["display_name","season"])
    master_rushing_data = master_rushing_data.drop(columns = ["season_type_x","week","player_position","team_abbr", "season_type_y","completions",
                                                              "attempts","passing_yards","passing_tds","interceptions","sacks",	"sack_yards","sack_fumbles","sack_fumbles_lost"
                                                              ,"passing_air_yards",	"passing_yards_after_catch","passing_first_downs","passing_epa","passing_2pt_conversions","pacr",
                                                              "dakota","receptions","targets","receiving_yards","receiving_tds","receiving_fumbles","receiving_fumbles_lost","receiving_air_yards",
                                                              "receiving_yards_after_catch","receiving_first_downs","receiving_epa","receiving_2pt_conversions","racr",	"target_share","air_yards_share",
                                                              "wopr_x","tgt_sh","ay_sh","yac_sh","wopr_y","ry_sh","rtd_sh",	"rfd_sh","rtdfd_sh","dom","w8dom","yptmpa","ppr_sh",	
                                                              "common_first_name","first_name","last_name",	"short_name"])
    master_rushing_data.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\master_rushing_data.csv")
    return master_rushing_data
def merge_receiving_data(seasonal_pfr_rec_data,ngs_receiving_data,merged_season_stats_player_data):
    merged_seasonal_pfr_ngs_rec_data = seasonal_pfr_rec_data.merge(ngs_receiving_data, left_on=["player","season"],right_on= ["player_display_name","season"],how="outer")
    # merged_seasonal_pfr_ngs_rec_data.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\test1.csv")
    master_receiving_data = merged_seasonal_pfr_ngs_rec_data.merge(merged_season_stats_player_data, how="right",left_on =["player","season"],right_on=["display_name","season"])
    master_receiving_data = master_receiving_data.drop(columns=["loaded","season_type_x","week","player_display_name","season_type_y","completions",
                                                              "attempts","passing_yards","passing_tds","interceptions","sacks",	"sack_yards","sack_fumbles","sack_fumbles_lost"
                                                              ,"passing_air_yards",	"passing_yards_after_catch","passing_first_downs","passing_epa","passing_2pt_conversions","pacr",
                                                              "dakota","carries","rushing_yards","rushing_tds",	"rushing_fumbles","rushing_fumbles_lost","rushing_first_downs",
                                                              "rushing_epa","rushing_2pt_conversions","first_name","last_name"])
    master_receiving_data.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\master_receiving_data.csv")
    return master_receiving_data
years = range(2014,2025)
fantasy_positions = ["QB", "RB", "TE", "WR"]
depth_chart = load_depth_chart_data(years, fantasy_positions)
season_stats = load_seasonal_data(years)
combine_data = load_combine_data(years, fantasy_positions)
draft_pick_data = load_draft_picks(years, fantasy_positions)
ngs_data_passing = load_ngs_data("passing",years)
ngs_data_rushing = load_ngs_data("rushing",years)
ngs_data_receiving = load_ngs_data("receiving",years)
player_data = load_player_data(years, fantasy_positions)
nfl_qbr_data = load_qbr_data(years, "nfl")
college_qbr_data = load_qbr_data(years,"college")
seasonal_pfr_pass_data = load_seasonal_pfr("pass")
seasonal_pfr_rush_data = load_seasonal_pfr("rush")
seasonal_pfr_rec_data = load_seasonal_pfr("rec")
snap_count_data = load_snap_counts(years, fantasy_positions)
# play_by_play_data = load_play_by_play_data(years)
merged_season_stats_player_data = merge_season_stats_player_data(season_stats,player_data)
master_passing_data = merge_passing_data(merged_season_stats_player_data,nfl_qbr_data,seasonal_pfr_pass_data,ngs_data_passing)
master_rushing_data = merge_rushing_data(seasonal_pfr_rush_data,ngs_data_rushing,merged_season_stats_player_data)
master_receiving_data = merge_receiving_data(seasonal_pfr_rec_data,ngs_data_receiving,merged_season_stats_player_data)