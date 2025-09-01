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
    #remove irrelevant columns like jersey number
    ngs_data = ngs_data.drop(columns="player_jersey_number")
    ngs_data.to_csv(fr"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\ngs_{stat_type}_data.csv")
    return ngs_data

years = range(2014,2025)
fantasy_positions = ["QB", "RB", "TE", "WR"]

depth_chart = load_depth_chart_data(years, fantasy_positions)
season_stats = load_seasonal_data(years)
combine_data = load_combine_data(years, fantasy_positions)
draft_pick_data = load_draft_picks(years, fantasy_positions)
ngs_data_passing = load_ngs_data("passing",years)
