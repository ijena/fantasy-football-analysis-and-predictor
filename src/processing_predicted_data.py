import pandas as pd 

predicted_df = pd.read_csv(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\predictions_2025\ALL_POSITIONS_MODEL_PROBS_2025.csv")

#renaming columns for clarity
predicted_df = predicted_df.rename(columns={'xgb_classification_prob_0':'xgb_classification_prob_neutral',
                                            'xgb_classification_prob_1' : 'xgb_classification_prob_over',
                                            'xgb_classification_prob_2' :'xgb_classification_prob_under',
                                            'xgb_percentile_prob_0'	:'xgb_percentile_prob_neutral',
                                            'xgb_percentile_prob_1' : 'xgb_percentile_prob_over',
                                            'xgb_percentile_prob_2': 'xgb_percentile_prob_under'
})

#calculating average probability for each classification across all models
predicted_df["average_probability_over"] = predicted_df[["xgb_classification_prob_over","xgb_percentile_prob_over",
                                                         "rf_classification_prob_over","rf_percentile_prob_over"]].mean(axis=1)

predicted_df["average_probability_under"] = predicted_df[["xgb_classification_prob_under","xgb_percentile_prob_under",
                                                         "rf_classification_prob_under","rf_percentile_prob_under"]].mean(axis=1)

predicted_df["average_probability_neutral"] = predicted_df[["xgb_classification_prob_neutral","xgb_percentile_prob_neutral",
                                                         "rf_classification_prob_neutral","rf_percentile_prob_neutral"]].mean(axis=1)

#convert all the averages to percentages and round it to 2 decimal places

predicted_df["average_probability_over"] = (predicted_df["average_probability_over"] * 100).round(2)
predicted_df["average_probability_under"] = (predicted_df["average_probability_under"] * 100).round(2)
predicted_df["average_probability_neutral"] = (predicted_df["average_probability_neutral"] * 100).round(2)

