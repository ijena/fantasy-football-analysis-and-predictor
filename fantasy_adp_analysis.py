import pandas as pd
import re
from pathlib import Path

# Load the 2015 ADP file
adp_path_2015 = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\data\Fantasy Pros ADP Data 2015-2024\FantasyPros_2015_Overall_ADP_Rankings.csv")
adp2015_raw = pd.read_csv(adp_path_2015)

# Clean name function 
def clean_name(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = re.sub(r'[\*\+]', ' ', s)  # remove * and +
    s = re.sub(r'\([^)]*\)', ' ', s)  # remove parentheses
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace(".", "").replace("'", "").lower()
    return s

# Identify player column
if "Player" in adp2015_raw.columns:
    player_col = "Player"
else:
    player_col = adp2015_raw.columns[1]

# Add cleaned columns
adp2015_raw['Player_fixed_2015'] = adp2015_raw[player_col].astype(str)
adp2015_raw['player_clean_2015'] = adp2015_raw['Player_fixed_2015'].apply(clean_name)

# Filter: only rows that look like real players (Firstname Lastname)
mask2015 = adp2015_raw['Player_fixed_2015'].str.contains(r'^[A-Za-z\.\'\-]+\s+[A-Za-z\.\'\-]+', regex=True, na=False)
adp2015 = adp2015_raw[mask2015].copy()

drop_cols = ["Team", "Bye", "Sleeper", "ESPN","NFL", "FFC","RTSports", "Fantrax"]

adp2015_cleaned = adp2015.drop(columns=[c for c in drop_cols if c in adp2015.columns])
# Create a new column with only the position letters (strip numbers)
adp2015_cleaned['POS_group'] = adp2015_cleaned['POS'].str.extract(r'([A-Za-z]+)')


print(adp2015_cleaned)



