import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    _HAS_LOESS = True
except Exception:
    _HAS_LOESS = False

def _fit_lookup(x_train, y_train, x_pred, loess_frac=0.3, min_group=25, fallback_bins=12):
    """Fit LOWESS on (x_train,y_train) and predict at x_pred; fallback to binned interpolation."""
    x_train = np.asarray(x_train); y_train = np.asarray(y_train)
    ok = ~np.isnan(x_train) & ~np.isnan(y_train)
    x_train, y_train = x_train[ok], y_train[ok]
    if len(x_train) < max(min_group, 5):
        return np.full_like(x_pred, np.nan, dtype=float)
    if _HAS_LOESS and len(x_train) >= min_group:
        try:
            smoothed = sm.nonparametric.lowess(y_train, x_train, frac=loess_frac, return_sorted=True)
            xs, ys = smoothed[:,0], smoothed[:,1]
            return np.interp(x_pred, xs, ys, left=np.nan, right=np.nan)
        except Exception:
            pass
    # Fallback: quantile bins on ADP -> mean PPG -> interpolate
    ranks = np.argsort(np.argsort(x_train))
    r = ranks / max(len(x_train) - 1, 1)
    try:
        q = pd.qcut(r, q=min(fallback_bins, len(x_train)), duplicates="drop")
        bin_means = pd.Series(y_train).groupby(q).mean()
        bin_adp   = pd.Series(x_train).groupby(q).median()
        xs, ys = bin_adp.values, bin_means.values
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        return np.interp(x_pred, xs, ys, left=np.nan, right=np.nan)
    except Exception:
        return np.full_like(x_pred, np.nan, dtype=float)

def add_expected_from_adp_history(
    df,
    *,
    year_col="year",
    adp_col="AVG",
    ppr_col="Fantasy_PPR",
    games_col="Games_G",
    pos_col="FantPos",
    loess_frac=0.3,
    min_group=25,
    fallback_bins=12,
):
    """
    Adds two leak-safe expected-PPR features:
      - expected_ppr_pg_prev: expectation computed in season t-1 and aligned to season t
      - expected_ppr_pg_curr_hist: expectation for season t using ONLY data from years < t
    Keeps your current-season actuals intact for labels.
    """
    df = df.copy()
    # numeric & ppg
    for c in [adp_col, ppr_col, games_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")
    df[adp_col] = pd.to_numeric(df[adp_col], errors="coerce")
    df[ppr_col] = pd.to_numeric(df[ppr_col], errors="coerce")
    df[games_col] = pd.to_numeric(df[games_col], errors="coerce")
    df["ppr_pg"] = np.where(df[games_col] > 0, df[ppr_col] / df[games_col], np.nan)

    # --- (A) Build expectations for each YEAR using ONLY prior years (historical fit) ---
    exp_curr_hist = pd.Series(np.nan, index=df.index, dtype=float)

    # Group by position, then iterate years ascending
    for pos, gpos in df.groupby(pos_col):
        gpos = gpos.sort_values(year_col)
        years = gpos[year_col].dropna().unique()
        for y in years:
            train_mask = gpos[year_col] < y     # strictly prior years
            pred_mask  = gpos[year_col] == y
            if not train_mask.any() or not pred_mask.any():
                continue
            x_train = gpos.loc[train_mask, adp_col].values
            y_train = gpos.loc[train_mask, "ppr_pg"].values
            x_pred  = gpos.loc[pred_mask, adp_col].values
            preds = _fit_lookup(x_train, y_train, x_pred,
                                loess_frac=loess_frac,
                                min_group=min_group,
                                fallback_bins=fallback_bins)
            exp_curr_hist.loc[gpos.loc[pred_mask].index] = preds

    df["expected_ppr_pg_curr_hist"] = exp_curr_hist

    # --- (B) Previous-season expectations: fit in year t-1 and align to year t ---
    # Compute expected per group (year, pos) using within-year fit, then shift +1 year to align
    exp_prev = pd.Series(np.nan, index=df.index, dtype=float)

    for (y, pos), g in df.groupby([year_col, pos_col]):
        x_tr = g[adp_col].values
        y_tr = g["ppr_pg"].values
        # predict on the same group's ADPs (these belong to year y)
        preds_y = _fit_lookup(x_tr, y_tr, x_tr,
                              loess_frac=loess_frac,
                              min_group=min_group,
                              fallback_bins=fallback_bins)
        # assign to the rows in year y+1 (same players will be joined via merge after shifting year)
        exp_prev.loc[g.index] = preds_y

    # shift expectations forward one season so they serve as "prev" for year t
    df["_exp_prev_tmp"] = exp_prev
    # Build a small (player, year+1) aligner
    align = df[[year_col, "player_clean", "_exp_prev_tmp"]].copy()
    align[year_col] = align[year_col] + 1
    df = df.merge(align.rename(columns={"_exp_prev_tmp": "expected_ppr_pg_prev"}),
                  on=["player_clean", year_col], how="left")
    df.drop(columns=["_exp_prev_tmp"], errors="ignore", inplace=True)

    return df

import numpy as np
import pandas as pd

SEASON_LEN = {y: (16 if y <= 2020 else 17) for y in range(2000, 2031)}

def add_expected_from_adp_history_with_season(
    df,
    *,
    year_col="year",
    adp_col="AVG",
    ppr_col="Fantasy_PPR",
    games_col="Games_G",
    pos_col="FantPos",
    loess_frac=0.3,
    min_group=25,
    fallback_bins=12,
):
    """
    Adds leak-safe expected PPR features (per-game + full season):
      - expected_ppr_pg_prev, expected_ppr_season_prev
      - expected_ppr_pg_curr_hist, expected_ppr_season_curr_hist
    Also adds:
      - ppr_pg (actual per game)
      - actual_ppr_season_std (actual per-game × era season length)
    """
    df = add_expected_from_adp_history(  # <-- the function from my previous message
        df,
        year_col=year_col,
        adp_col=adp_col,
        ppr_col=ppr_col,
        games_col=games_col,
        pos_col=pos_col,
        loess_frac=loess_frac,
        min_group=min_group,
        fallback_bins=fallback_bins,
    )

    # Era season length (default to 17 if year missing/out of range)
    season_len = df[year_col].map(SEASON_LEN).fillna(17)

    # Expected full-season totals (no leakage: both are derived without using year t outcomes)
    if "expected_ppr_pg_prev" in df:
        df["expected_ppr_season_prev"] = df["expected_ppr_pg_prev"] * season_len
    else:
        df["expected_ppr_season_prev"] = np.nan

    if "expected_ppr_pg_curr_hist" in df:
        df["expected_ppr_season_curr_hist"] = df["expected_ppr_pg_curr_hist"] * season_len
    else:
        df["expected_ppr_season_curr_hist"] = np.nan

    # Actual standardized season total (for labels/analysis, not a feature)
    # Requires ppr_pg (added by add_expected_from_adp_history via ppr_col/games_col)
    if "ppr_pg" in df:
        df["actual_ppr_season_std"] = df["ppr_pg"] * season_len
    else:
        df["actual_ppr_season_std"] = np.nan

    return df

from pathlib import Path
BASE = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data")
MERGED = BASE / "merged_dataset" / "merged_adp_ranks_2015_2024_with_metrics.csv"
OUT    = BASE / "merged_dataset" / "merged_with_expected_prev_and_currhist.csv"

df = pd.read_csv(MERGED)
df = add_expected_from_adp_history(
    df,
    year_col="year",
    adp_col="AVG",
    ppr_col="Fantasy_PPR",
    games_col="Games_G",
    pos_col="FantPos",
    loess_frac=0.3,
    min_group=25,
)

# What you now have:
# - expected_ppr_pg_prev        -> from season t-1 (aligned to t)  ✅ feature
# - expected_ppr_pg_curr_hist   -> for season t from past years    ✅ feature
# - Your same-season actuals (Fantasy_PPR, ppr_pg, etc.)           ✅ labels only

df.to_csv(OUT, index=False)
print("Saved:", OUT)

OUT    = BASE / "merged_dataset" / "merged_with_expected_pg_and_season.csv"

df = pd.read_csv(MERGED)

df = add_expected_from_adp_history_with_season(
    df,
    year_col="year",
    adp_col="AVG",
    ppr_col="Fantasy_PPR",
    games_col="Games_G",
    pos_col="FantPos",
    loess_frac=0.3,
    min_group=25,
)

df.to_csv(OUT, index=False)
print("Saved:", OUT)