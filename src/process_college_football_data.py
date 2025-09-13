import pandas as pd 

file_path = r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\college_football_data 2014-2024\cfbd_player_season_stats_2014_2024_master.csv"

college_football_data = pd.read_csv(file_path)

relevant_stat_categories = ["passing","rushing","receiving","interceptions","fumbles"]
college_football_data = college_football_data[college_football_data["category"].isin(relevant_stat_categories)]
college_football_data.to_csv(file_path)