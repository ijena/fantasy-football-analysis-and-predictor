import pandas as pd
import re
from pathlib import Path

# ---------- Config ----------
base_dir = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\Fantasy Pros ADP Data 2015-2024")

# Output folder directly under \data\
out_dir = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\cleaned_adp")
out_dir.mkdir(exist_ok=True, parents=True)

YEARS = range(2015, 2026)
file_pattern = "FantasyPros_{year}_Overall_ADP_Rankings.csv"

# Columns we don't need for modeling/merging
DROP_COLS = ["Team", "Bye", "ESPN", "Sleeper", "NFL", "RTSports", "FFC", "Fantrax", "CBS"]

# ---------- Helpers ----------
def clean_name(x: str) -> str:
    """Normalize player names for joining across sources."""
    if pd.isna(x):
        return ""
    s = str(x)
    s = re.sub(r"[\*\+]", " ", s)        # remove * and +
    s = re.sub(r"\([^)]*\)", " ", s)     # remove anything in parentheses
    s = re.sub(r"\s+", " ", s).strip()   # collapse whitespace
    s = s.replace(".", "").replace("'", "").lower()
    return s

def detect_player_col(df: pd.DataFrame) -> str:
    if "Player" in df.columns:
        return "Player"
    # Fallback: many FP files put "Player" second
    return df.columns[1] if len(df.columns) > 1 else df.columns[0]

def detect_pos_col(df: pd.DataFrame):
    # If you *know* it's always "POS", you could just: return "POS"
    for c in ["POS", "Pos", "Position"]:
        if "POS" in df.columns:
            return c
    return None

def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read with a couple of encodings to survive weird files."""
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="latin-1")

# ---------- Processing ----------
all_years = []

for year in YEARS:
    path = base_dir / file_pattern.format(year=year)
    if not path.exists():
        print(f"⚠️ Missing file for {year}: {path.name}")
        continue

    adp_raw = safe_read_csv(path)

    player_col = detect_player_col(adp_raw)
    pos_col = detect_pos_col(adp_raw)

    # Preserve original + cleaned names
    adp_raw["Player_fixed"] = adp_raw[player_col].astype(str)
    adp_raw["player_clean"] = adp_raw["Player_fixed"].apply(clean_name)

    # Keep rows that look like a "Firstname Lastname"
    mask = adp_raw["Player_fixed"].str.contains(
        r"^[A-Za-z\.\'\-]+\s+[A-Za-z\.\'\-]+", regex=True, na=False
    )
    df = adp_raw.loc[mask].copy()

    # Drop non-essential columns if present
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing)

    # POS_group (strip any trailing numbers, e.g., WR1 -> WR)
    if pos_col is not None:
        df["POS_group"] = df[pos_col].astype(str).str.extract(r"([A-Za-z]+)")
    else:
        df["POS_group"] = pd.NA

    df["year"] = year

    # convert AVG (average draft pick) to numeric
    df["AVG"] = pd.to_numeric(df.get("AVG"), errors="coerce")

    # ---- Overall ADP rank & percentile (within year) ----
    df["adp_rank"] = df["AVG"].rank(method="dense", ascending=True).astype("Int64")
    n_rank = int(df["adp_rank"].max()) if df["adp_rank"].notna().any() else 0

    if n_rank > 1:
        df["adp_percentile"] = 1 - ((df["adp_rank"] - 1) / (n_rank - 1))
    else:
        df["adp_percentile"] = 0.0
    df["adp_percentile"] = df["adp_percentile"].clip(0, 1)

    # ---- Position ADP rank & percentile (within POS_group, same year) ----
    # rank within each position: 1 = earliest pick in that position
    df["adp_rank_pos"] = (
        df.groupby("POS_group", dropna=False)["AVG"]
          .rank(method="dense", ascending=True)
          .astype("Int64")
    )

    # n per position (use max rank in group which equals group size for dense ranking)
    df["_n_pos"] = df.groupby("POS_group", dropna=False)["adp_rank_pos"].transform("max")

    # percentile within position: 1.0 = earliest pick in that position
    has_pos = df["_n_pos"].fillna(0) > 1
    df.loc[has_pos, "adp_percentile_pos"] = 1 - ((df.loc[has_pos, "adp_rank_pos"] - 1) / (df.loc[has_pos, "_n_pos"] - 1))
    df.loc[~has_pos, "adp_percentile_pos"] = 0.0  # edge case: 0 or 1 player in that POS
    df["adp_percentile_pos"] = df["adp_percentile_pos"].astype(float).clip(0, 1)

    # clean helper
    df = df.drop(columns=["_n_pos"])

    # Save per-year cleaned file
    out_path = out_dir / f"FantasyPros_ADP_cleaned_{year}.csv"
    df.to_csv(out_path, index=False)
    print(
        f"✅ Saved {year} → {out_path.name} "
        f"({df.shape[0]} rows, {df.shape[1]} cols) | "
        f"overall pct min/max: {df['adp_percentile'].min():.3f}/{df['adp_percentile'].max():.3f} | "
        f"pos pct min/max: {df['adp_percentile_pos'].min():.3f}/{df['adp_percentile_pos'].max():.3f}"
    )

    all_years.append(df)

# ---------- Master ----------
if all_years:
    master = pd.concat(all_years, ignore_index=True)
    master_out = out_dir / "FantasyPros_ADP_cleaned_2015_2024_master.csv"
    master.to_csv(master_out, index=False)
    print(
        f"\n🎉 Master dataset saved → {master_out} "
        f"({master.shape[0]} rows, {master.shape[1]} cols)"
    )
    print(
        f"   Global overall pct min/max: "
        f"{master['adp_percentile'].min():.3f}/{master['adp_percentile'].max():.3f}"
    )
    print(
        f"   Global pos pct min/max: "
        f"{master['adp_percentile_pos'].min():.3f}/{master['adp_percentile_pos'].max():.3f}"
    )
else:
    print("No years processed — check file names/paths.")
