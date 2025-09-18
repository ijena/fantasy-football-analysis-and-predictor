import re
from pathlib import Path
import numpy as np
import pandas as pd

# ========= USER PATHS (edit if needed) =========
BASE_DIR = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data")
INPUT_DIR = BASE_DIR / r"Pro Football Reference Fantasy Rank 2014-2024"
OUTPUT_DIR = BASE_DIR / "cleaned_fantasy_ranks"   # per-year + master outputs
# ==============================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
def clean_name(x: str) -> str:
    """Normalized name for joining (lowercase, strip punctuation, remove * + and parentheses)."""
    if pd.isna(x):
        return ""
    s = str(x)
    s = re.sub(r'[\*\+]', ' ', s)           # remove * and +
    s = re.sub(r'\([^)]*\)', ' ', s)        # remove parenthetical notes
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace(".", "").replace("'", "").lower()
    return s

def player_fixed(s: str) -> str:
    """Readable cleaned name (keep case; remove * + and parentheses; trim)."""
    if pd.isna(s):
        return ""
    t = re.sub(r'[\*\+]', ' ', str(s))
    t = re.sub(r'\([^)]*\)', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    PFR exports sometimes have 2 header rows (e.g., section + field).
    Flatten to single row and strip 'Unnamed' noise.
    """
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for a, b in df.columns:
            a = "" if pd.isna(a) or str(a).startswith("Unnamed") else str(a).strip()
            b = "" if pd.isna(b) or str(b).startswith("Unnamed") else str(b).strip()
            col = f"{a}_{b}".strip("_")
            new_cols.append(col if col else "col")
        df.columns = new_cols
    else:
        df.columns = [
            ("" if str(c).startswith("Unnamed") else str(c)).strip()
            for c in df.columns
        ]
    return df

def detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column (case-sensitive first, then case-insensitive)."""
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
    return None

def _percentile_from_rank(r: pd.Series) -> pd.Series:
    """
    Convert a 1=best rank Series to a percentile in [0,1],
    where 1.0 = best rank and 0.0 = worst. Missing ranks -> NaN.
    """
    r = pd.to_numeric(r, errors="coerce")     # ensure numeric
    n = pd.to_numeric(r, errors="coerce").max()
    out = pd.Series(np.nan, index=r.index, dtype=float)

    if pd.isna(n) or n <= 1:
        return out

    mask = r.notna()
    out.loc[mask] = 1.0 - ((r.loc[mask] - 1.0) / (n - 1.0))
    return out.clip(0.0, 1.0)

def add_helpers(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Add:
      - Player_fixed, player_clean
      - team (normalized), POS_group (letters only), year
      - PPR_rank_year (overall rank by PPR within the file/year)
      - ppr_percentile_overall (overall percentile from rank)
      - ppr_percentile_pos (percentile within POS_group)
    Keeps all original columns.
    """
    # Likely column names across PFR exports
    player_col = detect_col(df, ["Player", "Player_", "Player__"]) or df.columns[0]
    team_col   = detect_col(df, ["Tm", "Team"])
    pos_col    = detect_col(df, ["FantPos", "Pos", "POS"])
    ppr_col    = detect_col(df, ["PPR", "Fantasy_PPR", "Fantasy PPR", "PPR_"])

    # Names
    df["Player_fixed"] = df[player_col].apply(player_fixed)
    df["player_clean"] = df["Player_fixed"].apply(clean_name)

    # POS + POS_group
    if pos_col is not None:
        if "POS" not in df.columns:
            df["POS"] = df[pos_col]
        df["POS_group"] = (
            df["POS"].astype(str).str.extract(r"([A-Za-z]+)", expand=False).str.upper()
        )
    else:
        df["POS_group"] = pd.NA

    # Team normalized (keep original too)
    if team_col is not None:
        df["team"] = df[team_col].astype(str).str.upper().str.strip()
    else:
        df["team"] = pd.NA

    # Year
    df["year"] = int(year)

    # PPR rank within the year/file
    if ppr_col is None:
        # Try fuzzy match if headers were flattened oddly
        ppr_like = [c for c in df.columns if re.search(r"\bPPR\b", str(c), flags=re.I)]
        ppr_col = ppr_like[0] if ppr_like else None

    if ppr_col is not None:
        df["_ppr_num"] = pd.to_numeric(df[ppr_col], errors="coerce")
        df["PPR_rank_year"] = df["_ppr_num"].rank(method="min", ascending=False)
        df.drop(columns=["_ppr_num"], inplace=True)
    else:
        df["PPR_rank_year"] = np.nan

    # Percentiles (overall and within position)
    df["PPR_rank_year"] = pd.to_numeric(df["PPR_rank_year"], errors="coerce")
    df["ppr_percentile_overall"] = _percentile_from_rank(df["PPR_rank_year"])

    # Within-position percentile (compute rank per POS_group, then map to overall percentile logic)
    if "POS_group" in df.columns:
        # derive per-position rank
        pos_rank = (
            df.groupby("POS_group", dropna=False)["PPR_rank_year"]
              .rank(method="min", ascending=True)
        )
        # maximum rank per position (needed for normalization)
        pos_max = (
            df.groupby("POS_group", dropna=False)["PPR_rank_year"]
              .transform("max")
        )
        out = pd.Series(np.nan, index=df.index, dtype=float)
        mask = pos_rank.notna() & pos_max.notna() & (pos_max > 1)
        out.loc[mask] = 1.0 - ((pos_rank.loc[mask] - 1.0) / (pos_max.loc[mask] - 1.0))
        df["ppr_percentile_pos"] = out.clip(0.0, 1.0)
    else:
        df["ppr_percentile_pos"] = np.nan

    return df

def load_one_file(path: Path) -> pd.DataFrame:
    """Load a single PFR xlsx, flatten headers, drop empty 'Unnamed' cols."""
    # Try 2-row header first, fall back to 1-row
    try:
        df = pd.read_excel(path, header=[0, 1])
        df = flatten_columns(df)
    except Exception:
        df = pd.read_excel(path, header=0)
        df = flatten_columns(df)

    # Drop completely empty columns
    keep = [c for c in df.columns if not (str(c).startswith("Unnamed") or str(c).strip() == "")]
    df = df[keep]
    return df

def extract_year_from_name(name: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", name)
    return int(m.group(0)) if m else None

# ---------- Main ----------
def main():
    files = sorted(INPUT_DIR.glob("Fantasy Rank *.xlsx"))
    if not files:
        raise FileNotFoundError(f"No files found in: {INPUT_DIR}")

    per_year_frames = []
    for f in files:
        year = extract_year_from_name(f.name)
        # Keep only 2015–2024
        if year is None or not (2014 <= year <= 2024):
            print(f"Skipping {f.name} (year={year})")
            continue

        print(f"Processing {f.name} ...")
        df_raw = load_one_file(f)
        df_clean = add_helpers(df_raw, year=year)

        # Save per-year cleaned
        stub = f"fantasy_ranks_cleaned_{year}"
        csv_path = OUTPUT_DIR / f"{stub}.csv"
        parquet_path = OUTPUT_DIR / f"{stub}.parquet"

        df_clean.to_csv(csv_path, index=False)
        try:
            df_clean.to_parquet(parquet_path, index=False)
        except Exception:
            pass

        print(
            f"  Saved {csv_path.name} | "
            f"overall pct min/max: {df_clean['ppr_percentile_overall'].min(skipna=True):.3f}/"
            f"{df_clean['ppr_percentile_overall'].max(skipna=True):.3f} | "
            f"pos pct min/max: {df_clean['ppr_percentile_pos'].min(skipna=True):.3f}/"
            f"{df_clean['ppr_percentile_pos'].max(skipna=True):.3f}"
        )

        per_year_frames.append(df_clean)

    # Build master
    if not per_year_frames:
        raise RuntimeError("No yearly frames produced. Check file names and years.")

    master = pd.concat(per_year_frames, axis=0, ignore_index=True, sort=False)

    # Save master
    master_csv = OUTPUT_DIR / "fantasy_ranks_master_2014_2024.csv"
    master_parquet = OUTPUT_DIR / "fantasy_ranks_master_2014_2024.parquet"
    master.to_csv(master_csv, index=False)
    try:
        master.to_parquet(master_parquet, index=False)
    except Exception:
        pass

    print("\n✅ Done.")
    print(f"Per-year files → {OUTPUT_DIR}")
    print(f"Master CSV     → {master_csv}")
    if master_parquet.exists():
        print(f"Master Parquet → {master_parquet}")

if __name__ == "__main__":
    pd.options.display.width = 160
    pd.options.display.max_columns = 200
    main()
