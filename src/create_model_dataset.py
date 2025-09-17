import pandas as pd
import numpy as np

def clean_features(df, corr_threshold=0.99):
    """
    Cleans a fantasy football dataset by:
      1. Dropping exact duplicate columns
      2. Optionally dropping highly correlated columns

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset
    corr_threshold : float, optional (default=0.99)
        Threshold for considering features as duplicate
    drop_corr : bool, optional (default=True)
        Whether to drop highly correlated features automatically

    Returns
    -------
    df_cleaned : pandas.DataFrame
        Dataset with duplicate and correlated features removed
    dropped_columns : dict
        Dictionary with lists of dropped exact duplicates and correlations
    """
    
    dropped_columns = {"duplicates": [], "correlated": []}

    # --- Step 1: Drop exact duplicate columns ---
    duplicates = df.T.duplicated()
    df_cleaned = df.loc[:, ~duplicates]

    corr_matrix = df_cleaned.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_cols = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    df_cleaned = df_cleaned.drop(columns=corr_cols)
    
    return df_cleaned

master_passing_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\master_passing_data.csv").drop(columns=["Unnamed: 0"])
cleaned_master_passing_df = clean_features(master_passing_df)
master_receiving_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\master_receiving_data.csv")
cleaned_master_receiving_df = clean_features(master_receiving_df)
master_rushing_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\master_rushing_data.csv")
cleaned_master_rushing_df = clean_features(master_rushing_df)
#merging college qbr and counting stats
college_qbr_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\college_qbr_data.csv")
college_stats_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\college_football_data 2014-2024\cleaned_wide_cfbd_player_season_stats_2014_2024_master.csv")

adp_master_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\cleaned_adp\FantasyPros_ADP_cleaned_2015_2024_master.csv")
# Merge on season + player vs name_short
master_college_stats = pd.merge(
    college_stats_df,
    college_qbr_df,
    left_on=["season", "player"],
    right_on=["season", "name_short"],
    how="left")
cleaned_master_college_stats = clean_features(master_college_stats)
snap_counts_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\season_snap_count_data.csv")
combine_data_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\combine_data.csv")

merged_fantasy_rank_adp_with_expected_points_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\merged_dataset\merged_with_expected_pg_and_season.csv")
feature_cols = [
    "AVG", "adp_rank", "adp_percentile", "adp_percentile_pos",
    "POS_group", "FantPos", "year",'Player_fixed',
    "expected_ppr_pg_prev", "expected_ppr_season_prev",
    "expected_ppr_pg_curr_hist", "expected_ppr_season_curr_hist",
    "Games_G"   # if you want a proxy for durability
]

merged_expected_points_adp_df = merged_fantasy_rank_adp_with_expected_points_df[feature_cols]

qb_passing_rushing_df = cleaned_master_passing_df.merge(cleaned_master_rushing_df[cleaned_master_rushing_df["pos"]=="QB"],how="left",left_on = ["name_display","season"],right_on=["display_name","season"])
#qb_passing_rushing_df.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\test.csv")
qb_snap_count_passing_rushing_df = qb_passing_rushing_df.merge(snap_counts_df[snap_counts_df["position"]=="QB"],how="left",left_on =["name_display","season"],right_on=["player","season"])
# qb_snap_count_passing_rushing_df.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\test.csv")

qb_expected_points_adp_snap_count_passing_rushing_df = qb_snap_count_passing_rushing_df.merge(merged_expected_points_adp_df[merged_expected_points_adp_df["POS_group"]=="QB"],how="inner",left_on=["name_display","season"], right_on=["Player_fixed","year"])
# qb_expected_points_adp_snap_count_passing_rushing_df.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\test.csv")

master_qb_vet_model_df = qb_expected_points_adp_snap_count_passing_rushing_df[qb_expected_points_adp_snap_count_passing_rushing_df["rookie_x"] ==0]
master_qb_vet_model_df.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\master_qb_vet_data.csv")

qb_rookie_combine_model_data_df = qb_expected_points_adp_snap_count_passing_rushing_df[qb_expected_points_adp_snap_count_passing_rushing_df["rookie_x"] ==1].merge(combine_data_df,how="left",left_on=["name_display","season"],right_on=["player_name",'season'])
qb_rookie_combine_model_data_df.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\test.csv")

merged_combine_data_fantasy_rank_adp_with_expected_points_df = merged_fantasy_rank_adp_with_expected_points_df.merge(combine_data_df,how='left',left_on=["Player_fixed","year"],
                                                                                                                     right_on=["player_name","season"])
merged_combine_data_fantasy_rank_adp_with_expected_points_df= merged_combine_data_fantasy_rank_adp_with_expected_points_df.drop(columns=["player_name","season"])
merged_snap_counts_combine_data_fantasy_rank_adp_with_expected_points_df = merged_combine_data_fantasy_rank_adp_with_expected_points_df.merge(snap_counts_df,how="left",left_on=["Player_fixed","year"],
                                                                                                                                 right_on=["player","season"])
merged_snap_counts_combine_data_fantasy_rank_adp_with_expected_points_df = merged_snap_counts_combine_data_fantasy_rank_adp_with_expected_points_df.drop(columns=["player","season"])

merged_college_stats_snap_counts_combine_data_fantasy_rank_adp_with_expected_points_df = merged_combine_data_fantasy_rank_adp_with_expected_points_df.merge(cleaned_master_college_stats,how="left",left_on=["Player_fixed","year"],
                                                                                                                                                            right_on=["player","season"])
merged_college_stats_snap_counts_combine_data_fantasy_rank_adp_with_expected_points_df = merged_college_stats_snap_counts_combine_data_fantasy_rank_adp_with_expected_points_df.drop(columns=["player","season"])
# merged_college_stats_snap_counts_combine_data_fantasy_rank_adp_with_expected_points_df.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\model_data\test.csv")

