import numpy as np
import pandas as pd

# Optional LOWESS
try:
    import statsmodels.api as sm
    _HAS_LOESS = True
except Exception:
    _HAS_LOESS = False

# NFL regular-season length by year (simplified; adjust if you include 2020 oddities/playoffs)
SEASON_LEN = {y: (16 if y <= 2020 else 17) for y in range(2000, 2031)}

def add_expected_ppr_per_game(
    df: pd.DataFrame,
    *,
    adp_col: str = "AVG",
    ppr_col: str | None = None,
    games_col: str | None = None,
    pos_col: str | None = None,
    per_year: bool = True,        # fit per YEAR+POSITION as you suggested
    loess_frac: float = 0.3,
    min_group: int = 25,
    fallback_bins: int = 12,
) -> pd.DataFrame:
    """
    Adds per-game normalization and expectations:
      - ppr_pg
      - expected_ppr_pg
      - overperf_pg
      - expected_ppr_season_std  (expected per-game × era season length)
      - actual_ppr_season_std    (actual per-game × era season length)
      - overperf_season_std
    """
    df = df.copy()

    # --- column detection ---
    if adp_col not in df.columns:
        raise ValueError(f"ADP column '{adp_col}' not in df.")

    if ppr_col is None:
        for c in ["Fantasy_PPR", "PPR", "PPR_total", "PPR Points", "Fantasy PPR"]:
            if c in df.columns:
                ppr_col = c; break
        if ppr_col is None:
            raise ValueError("Could not find a PPR column; pass ppr_col explicitly.")

    if games_col is None:
        for c in ["G", "Games", "GP"]:
            if c in df.columns:
                games_col = c; break
        if games_col is None:
            raise ValueError("Could not find a games column; pass games_col explicitly.")

    if pos_col is None:
        for c in ["POS_group_adp", "POS_group_rank", "POS_group", "POS", "FantPos"]:
            if c in df.columns:
                pos_col = c; break
        if pos_col is None:
            raise ValueError("Could not find a position column; pass pos_col explicitly.")

    if "year" not in df.columns and per_year:
        raise ValueError("Column 'year' required when per_year=True.")

    # --- ensure numeric ---
    df[adp_col] = pd.to_numeric(df[adp_col], errors="coerce")
    df[ppr_col] = pd.to_numeric(df[ppr_col], errors="coerce")
    df[games_col] = pd.to_numeric(df[games_col], errors="coerce")

    # --- actual PPR per game ---
    df["ppr_pg"] = np.where(df[games_col] > 0, df[ppr_col] / df[games_col], np.nan)

    # --- fit expected per game from ADP ---
    expected_pg = pd.Series(np.nan, index=df.index, dtype=float)
    group_keys = ["year", pos_col] if per_year else [pos_col]

    def _fit_group(g: pd.DataFrame) -> pd.Series:
        gg = g[[adp_col, "ppr_pg"]].dropna()
        if len(gg) < max(min_group, 5):
            return pd.Series(np.nan, index=g.index)

        x = gg[adp_col].values
        y = gg["ppr_pg"].values

        # LOWESS smoothing if available
        if _HAS_LOESS and len(gg) >= min_group:
            try:
                smoothed = sm.nonparametric.lowess(y, x, frac=loess_frac, return_sorted=True)
                xs, ys = smoothed[:, 0], smoothed[:, 1]
                vals = np.interp(g[adp_col], xs, ys, left=np.nan, right=np.nan)
                return pd.Series(vals, index=g.index)
            except Exception:
                pass

        # Fallback: quantile bins on ADP rank -> mean PPG -> interpolate
        try:
            ranks = np.argsort(np.argsort(x))
            gg = gg.assign(_r=ranks / max(len(x) - 1, 1))
            gg["_bin"] = pd.qcut(gg["_r"], q=min(fallback_bins, len(gg)), duplicates="drop")
            bin_means = gg.groupby("_bin")["ppr_pg"].mean()
            bin_adp = gg.groupby("_bin")[adp_col].median()
            xs, ys = bin_adp.values, bin_means.values
            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]
            vals = np.interp(g[adp_col], xs, ys, left=np.nan, right=np.nan)
            return pd.Series(vals, index=g.index)
        except Exception:
            return pd.Series(np.nan, index=g.index)

    for _, g in df.groupby(group_keys):
        expected_pg.loc[g.index] = _fit_group(g)

    df["expected_ppr_pg"] = expected_pg
    df["overperf_pg"] = df["ppr_pg"] - df["expected_ppr_pg"]

    # --- season-standardized versions (use era schedule length for comparability) ---
    # If year missing or out of dict, default to 17
    season_len = df.get("year", pd.Series(17, index=df.index)).map(SEASON_LEN).fillna(17)
    df["expected_ppr_season_std"] = df["expected_ppr_pg"] * season_len
    df["actual_ppr_season_std"]   = df["ppr_pg"] * season_len
    df["overperf_season_std"]     = df["actual_ppr_season_std"] - df["expected_ppr_season_std"]

    return df
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data")
MERGED = BASE / "merged_dataset" / "merged_adp_ranks_2015_2024_with_metrics.csv"
OUT    = BASE / "merged_dataset" / "merged_with_expected_ppg.csv"

df = pd.read_csv(MERGED)

df = add_expected_ppr_per_game(
    df,
    adp_col="AVG",
    ppr_col="Fantasy_PPR",
    games_col="Games_G",             # change if your file uses "Games" or "GP"
    pos_col="POS_group_adp",
    per_year=True,             # fit per YEAR+POSITION (as you asked)
    loess_frac=0.3,
    min_group=25,
)

df.to_csv(OUT, index=False)
print("Saved:", OUT)
