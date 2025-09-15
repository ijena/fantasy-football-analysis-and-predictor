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

master_passing_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\master_passing_data.csv")
cleaned_master_passing_df = clean_features(master_passing_df)
master_receiving_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\master_receiving_data.csv")
cleaned_master_receiving_df = clean_features(master_receiving_df)
master_rushing_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\master_rushing_data.csv")
cleaned_master_rushing_df = clean_features(master_rushing_df)
