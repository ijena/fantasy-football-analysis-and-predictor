import duckdb

# Connect (this will create fantasy.duckdb in your working directory)
con = duckdb.connect("fantasy.duckdb")

# Paths to your files
predictions_path = r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\predictions_2025\final_model_2025.csv"
history_path     = r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\historic_data\merged_historic_data.csv"

# Create / overwrite tables
con.execute(f"CREATE OR REPLACE TABLE predictions AS SELECT * FROM read_csv_auto('{predictions_path}')")
con.execute(f"CREATE OR REPLACE TABLE history AS SELECT * FROM read_csv_auto('{history_path}')")

print("Tables created ✅")

# Quick peek at data
print("\nPredictions sample:")
print(con.execute("SELECT * FROM predictions LIMIT 5").df())

print("\nHistory sample:")
print(con.execute("SELECT * FROM history LIMIT 5").df())
