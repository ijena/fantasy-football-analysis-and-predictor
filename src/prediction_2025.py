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