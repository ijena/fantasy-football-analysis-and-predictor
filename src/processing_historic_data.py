import pandas as pd

def keep_columns(df,columns=["Player_fixed","POS_group","merge_year","per_game_perf_rel_expectations","AVG_ADP","ppg_Fantasy_PPR"]):
    #function to keep certain columns in the dataframe df
    df = df.rename(columns={'AVG':'AVG_ADP'})
    return df[columns]
    
    
qb_historic_data = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\historic_data\qb_dataset_with_historic_performance.csv")
rb_historic_data = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\historic_data\rb_dataset_with_historic_performance.csv")
te_historic_data = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\historic_data\te_dataset_with_historic_performance.csv")
wr_historic_data = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\historic_data\wr_dataset_with_historic_performance.csv")

qb_historic_data = keep_columns(qb_historic_data)
rb_historic_data = keep_columns(rb_historic_data)
te_historic_data = keep_columns(te_historic_data)
wr_historic_data = keep_columns(wr_historic_data)

merged_historic_Data = pd.concat([qb_historic_data,rb_historic_data,te_historic_data,wr_historic_data],
                                   ignore_index=True)

merged_historic_Data.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\historic_data\merged_historic_data.csv")

# qb_historical_data.to_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\historic_data\qb_dataset_with_historic_performance.csv")