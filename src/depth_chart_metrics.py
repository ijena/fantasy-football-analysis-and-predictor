import pandas as pd 
import re
from pathlib import Path
import numpy as np

# file path for depth chart data
file_path = r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\nflverse_data\depth_chart.csv"

# Load ply by play data into a pandas DataFrame
df = pd.read_csv(file_path)


def clean_name(x: str) -> str:
    if pd.isna(x): return ""
    s = str(x)
    s = re.sub(r'[\*\+]', ' ', s)
    s = re.sub(r'\([^)]*\)', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace(".", "").replace("'", "").lower()
    return s

def pick_first(df: pd.DataFrame, candidates):
    """Return the first column from candidates that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
        # case-insensitive fallback
        lc = {str(col).lower(): col for col in df.columns}
        if c.lower() in lc:
            return lc[c.lower()]
    return None

def derive_depth_rank(row, rank_col, pos_slot_col):
    """
    Return an integer depth rank if possible.
    Priority:
      1) direct rank integer column (e.g., 'depth_team_order', 'depth_chart_order', 'position_depth')
      2) parse number from a slot string like 'WR3' / 'RB2'
      3) fallback to NaN
    """
    # 1) direct rank
    if rank_col and pd.notna(row.get(rank_col)):
        try:
            r = int(pd.to_numeric(row[rank_col], errors="coerce"))
            if r > 0:
                return r
        except Exception:
            pass

    # 2) parse from slot like 'WR3'
    if pos_slot_col and pd.notna(row.get(pos_slot_col)):
        m = re.search(r'(\d+)$', str(row[pos_slot_col]))
        if m:
            try:
                r = int(m.group(1))
                if r > 0:
                    return r
            except Exception:
                pass
    return np.nan

def normalize_position(s):
    if pd.isna(s): return np.nan
    x = str(s).upper().strip()
    # collapse common variants
    if x.startswith("QB"): return "QB"
    if x.startswith("RB"): return "RB"
    if x.startswith("WR"): return "WR"
    if x.startswith("TE"): return "TE"
    return x

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Load ----------
    df = pd.read_csv(IN_PATH, low_memory=False)

    # Soft-select columns we need (be flexible to schema differences)
    season_col   = pick_first(df, ["season", "Season"])
    week_col     = pick_first(df, ["week", "Week"])
    team_col     = pick_first(df, ["team", "posteam", "Team"])
    pos_col      = pick_first(df, ["position", "pos", "depth_chart_position", "depth_pos"])
    slot_col     = pick_first(df, ["depth_chart_position", "depth_chart_slot", "slot", "position_group"])
    rank_cols    = ["depth_team_order", "depth_chart_order", "position_depth", "depth"]
    rank_col     = pick_first(df, rank_cols)
    name_col     = pick_first(df, ["player_name", "full_name", "name", "gsis_name", "display_name"])
    pid_col      = pick_first(df, ["gsis_id", "player_id", "nfl_id", "pfr_player_id", "fantasy_player_id"])
    st_col       = pick_first(df, ["status", "roster_status"])  # e.g., "Active" / "Practice Squad" / "IR"
    st_type_col  = pick_first(df, ["season_type"])              # "Regular", "Post", etc.

    if season_col is None or week_col is None or team_col is None or pos_col is None:
        raise ValueError("Missing a required column among season/week/team/position. Inspect your file headers.")

    # Keep only regular season if available and desired
    if REG_ONLY and st_type_col and st_type_col in df.columns:
        df = df[df[st_type_col].astype(str).str.contains("Reg", case=False, na=False)]

    # ---------- Normalize core columns ----------
    df = df.copy()
    df.rename(columns={
        season_col: "season",
        week_col: "week",
        team_col: "team",
        pos_col: "position_raw",
    }, inplace=True)

    # fallback name/id
    if name_col and name_col in df.columns:
        df["player_name"] = df[name_col].astype(str)
    else:
        df["player_name"] = df.get("player", df.get("name", np.nan)).astype(str)

    if pid_col and pid_col in df.columns:
        df["player_id"] = df[pid_col].astype(str)
    else:
        df["player_id"] = np.nan

    if slot_col and slot_col in df.columns:
        df["slot_raw"] = df[slot_col].astype(str)
    else:
        df["slot_raw"] = np.nan

    if st_col and st_col in df.columns:
        df["status"] = df[st_col].astype(str)
    else:
        df["status"] = np.nan

    # POS normalized and filter to fantasy positions
    df["POS_group"] = df["position_raw"].apply(normalize_position)
    df = df[df["POS_group"].isin(KEEP_POS)].copy()

    # ---------- Depth rank ----------
    df["depth_rank"] = df.apply(lambda r: derive_depth_rank(r, rank_col, "slot_raw"), axis=1)
    # If still missing, attempt to infer rank=1 for slots that end with '1', else NaN
    df.loc[df["depth_rank"].isna() & df["slot_raw"].str.contains(r'\d$', na=False), "depth_rank"] = (
        df["slot_raw"].str.extract(r'(\d+)$')[0].astype(float)
    )

    # ---------- Starter flags ----------
    # Base "starter" = depth_rank == 1
    df["is_starter"] = (df["depth_rank"] == 1).astype(int)

    # WR helper flags (teams often start 2-3 WRs; this gives more nuance)
    df["is_wr_top2"] = np.where(df["POS_group"] == "WR", (df["depth_rank"] <= 2).astype(int), 0)
    df["is_wr_top3"] = np.where(df["POS_group"] == "WR", (df["depth_rank"] <= 3).astype(int), 0)

    # Cleaned name key for merging
    df["player_clean"] = df["player_name"].apply(clean_name)

    # ---------- Weekly tidy ----------
    weekly_cols = [
        "season", "week", "team", "player_id", "player_name", "player_clean",
        "POS_group", "position_raw", "slot_raw", "depth_rank", "is_starter",
        "is_wr_top2", "is_wr_top3", "status"
    ]
    weekly = df[weekly_cols].copy().sort_values(["season", "week", "team", "POS_group", "depth_rank"])

    # ---------- Season aggregation (stability, deltas) ----------
    # We compute per (season, team, player) metrics
    def add_rank_change_stats(g):
        g = g.sort_values("week")
        depth = g["depth_rank"].astype(float)
        changes = depth.diff().fillna(0).ne(0).sum()  # count changes across weeks
        # promotions = times rank number got smaller (e.g., 3 -> 2 or 2 -> 1)
        promotions = (depth.diff() < 0).sum()
        demotions  = (depth.diff() > 0).sum()
        return pd.Series({
            "weeks_listed": g.shape[0],
            "weeks_starter": g["is_starter"].sum(),
            "starter_rate": g["is_starter"].mean(),
            "avg_depth_rank": np.nanmean(depth),
            "min_depth_rank": np.nanmin(depth),
            "max_depth_rank": np.nanmax(depth),
            "rank_std": np.nanstd(depth),
            "rank_changes": int(changes),
            "promotions": int(promotions),
            "demotions": int(demotions),
            "first_week_seen": g["week"].min(),
            "last_week_seen": g["week"].max(),
            "week1_depth_rank": g.loc[g["week"].idxmin(), "depth_rank"] if g.shape[0] else np.nan,
            "final_week_depth_rank": g.loc[g["week"].idxmax(), "depth_rank"] if g.shape[0] else np.nan,
            "wr_top2_weeks": g["is_wr_top2"].sum(),
            "wr_top3_weeks": g["is_wr_top3"].sum(),
        })

    season_agg = (
        weekly
        .groupby(["season", "team", "player_id", "player_name", "player_clean", "POS_group"], dropna=False)
        .apply(add_rank_change_stats)
        .reset_index()
    )

    # Add “starter on opening day” flag
    season_agg["starter_week1"] = (season_agg["week1_depth_rank"] == 1).astype(int)

    # If a player appears for multiple teams in a season, you’ll have multiple rows.
    # That’s good: team context matters for joining with team-year features later.

    # ---------- Save ----------
    weekly.to_csv(OUT_WEEK, index=False)
    season_agg.to_csv(OUT_SEAS, index=False)

    print(f"✅ Saved weekly cleaned depth chart → {OUT_WEEK}  (rows={len(weekly)})")
    print(f"✅ Saved season aggregation → {OUT_SEAS}  (rows={len(season_agg)})")

    # ---------- (Optional) Quick sanity preview ----------
    print("\nTop 10 weekly rows:")
    print(weekly.head(10))
    print("\nSample season aggregates:")
    print(season_agg.sort_values(['season','team','POS_group','avg_depth_rank']).head(10))

if __name__ == "__main__":
    pd.options.display.width = 140
    pd.options.display.max_columns = 100
    main()

