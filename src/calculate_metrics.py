import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    _HAS_LOESS = True
except Exception:
    _HAS_LOESS = False


def add_expected_ppr(
    df: pd.DataFrame,
    adp_col: str = "AVG",
    ppr_col: str | None = None,
    pos_col: str | None = None,
    per_year: bool = False,            # <-- default to pooled
    loess_frac: float = 0.3,
    min_group: int = 25,
    fallback_bins: int = 12,
) -> pd.DataFrame:
    """
    Adds:
      - expected_PPR: pooled (position-level) smoothed ADP->PPR expectation
      - overperf_points: actual minus expected
    """
    if adp_col not in df.columns:
        raise ValueError(f"ADP column '{adp_col}' not found in df.")

    if ppr_col is None:
        for c in ["Fantasy_PPR", "PPR", "PPR_total", "PPR Points", "Fantasy PPR", "PPR_"]:
            if c in df.columns:
                ppr_col = c
                break
        if ppr_col is None:
            raise ValueError("Could not find a PPR column; please pass ppr_col explicitly.")

    if pos_col is None:
        for c in ["POS_group_adp", "POS_group_rank", "POS_group", "POS"]:
            if c in df.columns:
                pos_col = c
                break
        if pos_col is None:
            raise ValueError("Could not find a position column; please pass pos_col explicitly.")

    df = df.copy()
    df[adp_col] = pd.to_numeric(df[adp_col], errors="coerce")
    df[ppr_col] = pd.to_numeric(df[ppr_col], errors="coerce")

    expected = pd.Series(np.nan, index=df.index, dtype=float)

    # POOLED: one curve per position across all years
    group_keys = [pos_col] if not per_year else ["year", pos_col]

    def _fit_group(g: pd.DataFrame) -> pd.Series:
        gg = g[[adp_col, ppr_col]].dropna()
        if len(gg) < max(min_group, 5):
            return pd.Series(np.nan, index=g.index)

        x = gg[adp_col].values
        y = gg[ppr_col].values

        if _HAS_LOESS and len(gg) >= min_group:
            try:
                smoothed = sm.nonparametric.lowess(y, x, frac=loess_frac, return_sorted=True)
                xs, ys = smoothed[:, 0], smoothed[:, 1]
                exp_vals = np.interp(g[adp_col], xs, ys, left=np.nan, right=np.nan)
                return pd.Series(exp_vals, index=g.index)
            except Exception:
                pass

        try:
            gg = gg.assign(_rank=np.argsort(np.argsort(x)) / max(len(x) - 1, 1))
            gg["_bin"] = pd.qcut(gg["_rank"], q=min(fallback_bins, len(gg)), duplicates="drop")
            bin_means = gg.groupby("_bin")[ppr_col].mean()
            bin_adp = gg.groupby("_bin")[adp_col].median()
            xs = bin_adp.values
            ys = bin_means.values
            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]
            exp_vals = np.interp(g[adp_col], xs, ys, left=np.nan, right=np.nan)
            return pd.Series(exp_vals, index=g.index)
        except Exception:
            return pd.Series(np.nan, index=g.index)

    for keys, g in df.groupby(group_keys):
        expected.loc[g.index] = _fit_group(g)

    df["expected_PPR"] = expected
    df["overperf_points"] = df[ppr_col] - df["expected_PPR"]
    return df
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data")
MERGED_PATH = BASE / "merged_dataset" / "merged_adp_ranks_2015_2024_with_metrics.csv"
OUT_PATH    = BASE / "merged_dataset" / "merged_adp_ranks_2015_2024_with_expected_points.csv"

df = pd.read_csv(MERGED_PATH)

df = add_expected_ppr(
    df,
    adp_col="AVG",
    ppr_col="Fantasy_PPR",
    pos_col="POS_group_adp",   # or "POS_group_rank"
    per_year=False,            # <-- pooled across years by position
    loess_frac=0.3,
    min_group=25,
)

df.to_csv(OUT_PATH, index=False)
print("Saved pooled expected_PPR:", OUT_PATH)
