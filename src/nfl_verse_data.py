import nfl_data_py
import pandas as pd

def load_depth_chart_data(years):


    depth_charts = nfl_data_py.import_depth_charts(years)

    #filter for fantasy relevant positions
    fantasy_positions = ["QB", "RB", "TE", "WR"]
    
    depth_charts = depth_charts.drop(columns = ["jersey_number"])
    depth_charts = depth_charts[depth_charts["position"].isin(fantasy_positions)]
    #filter for regular season games only
    depth_charts = depth_charts[depth_charts["game_type"]=="REG"]
    #filter for offense
    depth_charts = depth_charts[depth_charts["formation"]=="Offense"]

    depth_charts.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\depth_chart.csv")   

def load_seasonal_data(years):
    season_stats = nfl_data_py.import_seasonal_data(years)
    season_stats = pd.DataFrame(season_stats)
    season_stats.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\season_stats.csv")   
years = range(2014,2025)
depth_chart = load_depth_chart_data(years)
load_seasonal_data(years)