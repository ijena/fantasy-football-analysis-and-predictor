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

# -------------------------------
# Create views
# -------------------------------

con.execute("""
CREATE OR REPLACE VIEW v_predictions AS
SELECT
  Player_fixed AS player,
  team_name             AS team,
  position                      AS position,
  merge_year                                        AS year,
  AVG_ADP                                           AS AVG_ADP,
  average_probability_over                          AS average_probability_over,
  average_probability_under                         AS average_probability_under,
  average_probability_neutral                       AS average_probability_neutral
FROM predictions;
""")

con.execute("""
CREATE OR REPLACE VIEW v_history AS
SELECT
  Player_fixed   AS player,
  POS_group                         AS position,
  merge_year                         AS year,
  per_game_perf_rel_expectations  AS ppg_diff,
  AVG_ADP                          AS AVG_ADP,
  ppg_Fantasy_PPR                 AS ppg_fantasy_ppr        
FROM history;
""")

con.execute("""
CREATE OR REPLACE VIEW v_adp AS
SELECT player, position, year, AVG_ADP::DOUBLE AS adp
FROM v_predictions
WHERE AVG_ADP IS NOT NULL
UNION ALL
SELECT player, position, year, AVG_ADP::DOUBLE AS adp
FROM v_history
WHERE AVG_ADP IS NOT NULL;
""")

print("Views created ✅")


