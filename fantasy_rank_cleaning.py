import re
from pathlib import Path
import pandas as pd

# ========= USER PATHS (edit if needed) =========
BASE_DIR = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data")
INPUT_DIR = BASE_DIR / r"Pro Football Reference Fantasy Rank 2014-2024"
OUTPUT_DIR = BASE_DIR / "cleaned_fantasy_ranks"  # new folder for per-year + master outputs
# ==============================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_name(x: str) -> str:
    """Normalize player names for merging (lowercase, strip punctuation, remove * + and () stuff)."""
    if pd.isna(x):
        return ""
    s = str(x)
    s = re.sub(r'[\*\+]', ' ', s)          # remove * and +
    s = re.sub(r'\([^)]*\)', ' ', s)       # remove parenthetical notes
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace(".", "").replace("'", "").lower()
    return s

def player_fixed(s: str) -> str:
    """Readable cleaned name (keep case but remove * + and parentheticals; trim)."""
    if pd.isna(s):
        return ""
    t = re.sub(r'[\*\+]', ' ', str(s))
    t = re.sub(r'\([^)]*\)', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    PFR Excel sometimes has 2 header rows (e.g., section headers like 'Passing' + field names).
    This flattens a MultiIndex header into single strings and strips 'Unnamed' noise.
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
            ("" if str(c).startswith("Unnamed") else str(c)).strip() for c in df.columns
        ]
    return df

def detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
    return None

def _percentile_from_rank(rank_series: pd.Series) -> pd.Series:
    """
    Convert rank (1 = best) to percentile in [0,1] where 1.0 = best.
    Uses (1 - (rank-1)/(n-1)) with safe handling for n <= 1.
    """
    r = pd.to_numeric(rank_series, errors="coerce")
    valid = r.notna()
    out = pd.Series(pd.NA, index=r.index, dtype="float")
    n = valid.sum()
    if n <= 1:
        out.loc[valid] = 1.0  # only one valid row -> give it 1.0
        return out
    out.loc[valid] = 1.0 - (r[valid] - 1.0) / (n - 1.0)
    # hard clip just in case
    return out.clip(0.0, 1.0)

def add_helpers(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Add Player_fixed, player_clean, team, POS_group, year,
    PPR_rank_year (overall rank by PPR within year),
    ppr_percentile_overall (0..1, 1=best),
    ppr_percentile_pos (0..1, 1=best within POS).
    Keeps all original columns intact.
    """
    # Identify likely column names
    player_col = detect_col(df, ["Player", "Player_", "Player__"]) or df.columns[0]
    team_col   = detect_col(df, ["Tm", "Team"])
    pos_col    = detect_col(df, ["FantPos", "Pos", "POS"])
    ppr_col    = detect_col(df, ["PPR", "Fantasy_PPR", "Fantasy PPR", "PPR_"])

    # Add readable + normalized names
    df["Player_fixed"] = df[player_col].apply(player_fixed)
    df["player_clean"] = df["Player_fixed"].apply(clean_name)

    # POS and POS_group
    if pos_col is not None:
        if "POS" not in df.columns:
            df["POS"] = df[pos_col]
        df["POS_group"] = (
            df["POS"].astype(str).str.extract(r"([A-Za-z]+)", expand=False).str.upper()
        )
    else:
        df["POS_group"] = pd.NA

    # Team normalized (don’t drop the original)
    if team_col is not None:
        df["team"] = df[team_col].astype(str).str.upper().str.strip()
    else:
        df["team"] = pd.NA

    # Year column
    df["year"] = int(year)

    # PPR column detection fallback
    if ppr_col is None:
        ppr_like = [c for c in df.columns if re.fullmatch(r".*\bPPR\b.*", str(c), flags=re.I)]
        ppr_col = ppr_like[0] if ppr_like else None

    # ---- Rankings & Percentiles ----
    if ppr_col is not None:
        df["_ppr_num"] = pd.to_numeric(df[ppr_col], errors="coerce")

        # Overall (within this file = within year) rank: 1 = best (higher PPR is better)
        df["PPR_rank_year"] = df["_ppr_num"].rank(method="min", ascending=False)

        # Percentile overall from rank
        df["ppr_percentile_overall"] = _percentile_from_rank(df["PPR_rank_year"])

        # Percentile by position (within same year & POS_group)
        def by_pos(group: pd.DataFrame) -> pd.DataFrame:
            # rank within the group (POS), 1 = best
            r = group["_ppr_num"].rank(method="min", ascending=False)
            group["ppr_percentile_pos"] = _percentile_from_rank(r)
            return group

        if "POS_group" in df.columns:
            df = df.groupby("POS_group", dropna=False, group_keys=False).apply(by_pos)
        else:
            df["ppr_percentile_pos"] = pd.NA

        df.drop(columns=["_ppr_num"], inplace=True)
    else:
        df["PPR_rank_year"] = pd.NA
        df["ppr_percentile_overall"] = pd.NA
        df["ppr_percentile_pos"] = pd.NA

    return df

def load_one_file(path: Path) -> pd.DataFrame:
    """Load a single PFR xlsx, flatten headers, and return as DataFrame."""
    # Try 2-row header first, then fall back to 1-row
    try:
        df = pd.read_excel(path, header=[0, 1])
        df = flatten_columns(df)
    except Exception:
        df = pd.read_excel(path, header=0)
        df = flatten_columns(df)
    # Drop completely empty/unnamed columns (common with wide exports)
    keep = [c for c in df.columns if not (str(c).startswith("Unnamed") or str(c).strip() == "")]
    df = df[keep]
    return df

def extract_year_from_name(name: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", name)
    return int(m.group(0)) if m else None

def main():
    files = sorted(INPUT_DIR.glob("Fantasy Rank *.xlsx"))
    if not files:
        raise FileNotFoundError(f"No files found in: {INPUT_DIR}")

    per_year_frames = []
    for f in files:
        year = extract_year_from_name(f.name)
        if year is None:
            print(f"⚠️  Skipping (no year in filename): {f.name}")
            continue

        print(f"Processing {f.name} ...")
        df_raw = load_one_file(f)
        df_clean = add_helpers(df_raw, year=year)

        # Save per-year cleaned
        year_stub = f"fantasy_ranks_cleaned_{year}"
        df_clean.to_csv(OUTPUT_DIR / f"{year_stub}.csv", index=False)
        try:
            df_clean.to_parquet(OUTPUT_DIR / f"{year_stub}.parquet", index=False)
        except Exception:
            # Parquet optional (pyarrow/fastparquet may not be installed)
            pass

        per_year_frames.append(df_clean)

    # Build master (concat all; keep all columns)
    master = pd.concat(per_year_frames, axis=0, ignore_index=True, sort=False)

    # Save master
    master_csv = OUTPUT_DIR / "fantasy_ranks_master_2015_2024.csv"
    master_parquet = OUTPUT_DIR / "fantasy_ranks_master_2015_2024.parquet"
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
