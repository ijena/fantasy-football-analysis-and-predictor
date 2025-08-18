# import pandas as pd
# import re
# from pathlib import Path

# # Load the 2015 ADP file
# adp_path_2015 = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\data\Fantasy Pros ADP Data 2015-2024\FantasyPros_2015_Overall_ADP_Rankings.csv")
# adp2015_raw = pd.read_csv(adp_path_2015)

# # Clean name function 
# def clean_name(x: str) -> str:
#     if pd.isna(x):
#         return ""
#     s = str(x)
#     s = re.sub(r'[\*\+]', ' ', s)  # remove * and +
#     s = re.sub(r'\([^)]*\)', ' ', s)  # remove parentheses
#     s = re.sub(r'\s+', ' ', s).strip()
#     s = s.replace(".", "").replace("'", "").lower()
#     return s

# # Identify player column
# if "Player" in adp2015_raw.columns:
#     player_col = "Player"
# else:
#     player_col = adp2015_raw.columns[1]

# # Add cleaned columns
# adp2015_raw['Player_fixed_2015'] = adp2015_raw[player_col].astype(str)
# adp2015_raw['player_clean_2015'] = adp2015_raw['Player_fixed_2015'].apply(clean_name)

# # Filter: only rows that look like real players (Firstname Lastname)
# mask2015 = adp2015_raw['Player_fixed_2015'].str.contains(r'^[A-Za-z\.\'\-]+\s+[A-Za-z\.\'\-]+', regex=True, na=False)
# adp2015 = adp2015_raw[mask2015].copy()

# drop_cols = ["Team", "Bye", "Sleeper", "ESPN","NFL", "FFC","RTSports", "Fantrax"]

# adp2015_cleaned = adp2015.drop(columns=[c for c in drop_cols if c in adp2015.columns])
# # Create a new column with only the position letters (strip numbers)
# adp2015_cleaned['POS_group'] = adp2015_cleaned['POS'].str.extract(r'([A-Za-z]+)')


# print(adp2015_cleaned)

import pandas as pd
import re
from pathlib import Path

# ---------- Config ----------
base_dir = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data")
out_dir = base_dir / "cleaned"
out_dir.mkdir(exist_ok=True)

YEARS = range(2015, 2025)
file_pattern = "FantasyPros_{year}_Overall_ADP_Rankings.csv"

# Columns to drop if present
DROP_COLS = ["Team", "CBS","Bye", "ESPN", "Sleeper", "NFL", "RTSports", "FFC", "Fantrax"]

# ---------- Helpers ----------
def clean_name(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = re.sub(r'[\*\+]', ' ', s)        # remove * and +
    s = re.sub(r'\([^)]*\)', ' ', s)     # remove parentheses and their content
    s = re.sub(r'\s+', ' ', s).strip()   # collapse whitespace
    s = s.replace(".", "").replace("'", "").lower()
    return s

def detect_player_col(df: pd.DataFrame) -> str:
    if "Player" in df.columns:
        return "Player"
    return df.columns[1] if len(df.columns) > 1 else df.columns[0]

def detect_pos_col(df: pd.DataFrame):
    for c in ["POS", "Pos", "Position"]:
        if c in df.columns:
            return c
    return None

# ---------- Processing ----------
all_years = []

for year in YEARS:
    path = base_dir / file_pattern.format(year=year)
    if not path.exists():
        print(f"⚠️ Missing file for {year}: {path.name}")
        continue

    try:
        adp_raw = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except UnicodeDecodeError:
        adp_raw = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="latin-1")

    player_col = detect_player_col(adp_raw)
    pos_col = detect_pos_col(adp_raw)

    # Preserve original, then cleaned names
    adp_raw["Player_fixed"] = adp_raw[player_col].astype(str)
    adp_raw["player_clean"] = adp_raw["Player_fixed"].apply(clean_name)

    # Keep only rows that look like Firstname Lastname
    mask = adp_raw["Player_fixed"].str.contains(r"^[A-Za-z\.\'\-]+\s+[A-Za-z\.\'\-]+", regex=True, na=False)
    df = adp_raw.loc[mask].copy()

    # Drop unwanted columns if they exist
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing)

    # POS_group (strip trailing numbers) if a pos column exists
    if pos_col is not None:
        df["POS_group"] = df[pos_col].astype(str).str.extract(r"([A-Za-z]+)")
    else:
        df["POS_group"] = pd.NA

    # Add year for later grouping/stacking
    df["year"] = year

    # Save per-year cleaned file
    out_path = out_dir / f"FantasyPros_ADP_cleaned_{year}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {year} → {out_path.name} ({df.shape[0]} rows, {df.shape[1]} cols)")

    all_years.append(df)

# Combine into one master file
if all_years:
    master = pd.concat(all_years, ignore_index=True)
    master_out = out_dir / "FantasyPros_ADP_cleaned_2015_2024_master.csv"
    master.to_csv(master_out, index=False)
    print(f"\n🎉 Master dataset saved → {master_out.name} ({master.shape[0]} rows, {master.shape[1]} cols)")
else:
    print("No years processed — check file names/paths.")
import pandas as pd
import re
from pathlib import Path

# ---------- Config ----------
base_dir = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\Fantasy Pros ADP Data 2015-2024")

# New output folder directly under \data\
out_dir = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\cleaned_adp")
out_dir.mkdir(exist_ok=True)

YEARS = range(2015, 2025)
file_pattern = "FantasyPros_{year}_Overall_ADP_Rankings.csv"

DROP_COLS = ["Team", "Bye", "ESPN", "Sleeper", "NFL", "RTSports", "FFC", "Fantrax"]

# ---------- Helpers ----------
def clean_name(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = re.sub(r'[\*\+]', ' ', s)        # remove * and +
    s = re.sub(r'\([^)]*\)', ' ', s)     # remove parentheses
    s = re.sub(r'\s+', ' ', s).strip()   # collapse whitespace
    s = s.replace(".", "").replace("'", "").lower()
    return s

def detect_player_col(df: pd.DataFrame) -> str:
    if "Player" in df.columns:
        return "Player"
    return df.columns[1] if len(df.columns) > 1 else df.columns[0]

def detect_pos_col(df: pd.DataFrame):
    for c in ["POS", "Pos", "Position"]:
        if c in df.columns:
            return c
    return None

# ---------- Processing ----------
all_years = []

for year in YEARS:
    path = base_dir / file_pattern.format(year=year)
    if not path.exists():
        print(f"⚠️ Missing file for {year}: {path.name}")
        continue

    try:
        adp_raw = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except UnicodeDecodeError:
        adp_raw = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="latin-1")

    player_col = detect_player_col(adp_raw)
    pos_col = detect_pos_col(adp_raw)

    adp_raw["Player_fixed"] = adp_raw[player_col].astype(str)
    adp_raw["player_clean"] = adp_raw["Player_fixed"].apply(clean_name)

    mask = adp_raw["Player_fixed"].str.contains(r"^[A-Za-z\.\'\-]+\s+[A-Za-z\.\'\-]+", regex=True, na=False)
    df = adp_raw.loc[mask].copy()

    drop_existing = [c for c in DROP_COLS if c in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing)

    if pos_col is not None:
        df["POS_group"] = df[pos_col].astype(str).str.extract(r"([A-Za-z]+)")
    else:
        df["POS_group"] = pd.NA

    df["year"] = year

    out_path = out_dir / f"FantasyPros_ADP_cleaned_{year}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {year} → {out_path.name} ({df.shape[0]} rows, {df.shape[1]} cols)")

    all_years.append(df)

# ---------- Master ----------
if all_years:
    master = pd.concat(all_years, ignore_index=True)
    master_out = out_dir / "FantasyPros_ADP_cleaned_2015_2024_master.csv"
    master.to_csv(master_out, index=False)
    print(f"\n🎉 Master dataset saved → {master_out} ({master.shape[0]} rows, {master.shape[1]} cols)")
else:
    print("No years processed — check file names/paths.")
