import os
import time
from pathlib import Path
import pandas as pd

from dotenv import load_dotenv
import cfbd
from cfbd.rest import ApiException

# ----------------- Config -----------------
YEARS = range(2014, 2025)  # 2014–2024 inclusive
SAVE_DIR = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\college_football_data 2014-2024")  # <- change this

FANTASY_POS = {"QB", "RB", "WR", "TE"}  # optional filter
ONLY_FANTASY_POSITIONS = True         # set True to keep only those positions

# ----------------- Auth -------------------
load_dotenv()
token = os.getenv("BEARER_TOKEN")  # <- put BEARER_TOKEN=yourkey in .env (NO quotes)
if not token:
    raise RuntimeError("BEARER_TOKEN missing. Put it in your .env file as BEARER_TOKEN=...")

# Configure cfbd to use Bearer token (Authorization: Bearer <token>)
configuration = cfbd.Configuration(
    host="https://api.collegefootballdata.com",
    access_token=token
)

# ----------------- Helpers ----------------
def rows_to_df(rows):
    """cfbd returns model objects—convert robustly to DataFrame."""
    if not rows:
        return pd.DataFrame()
    def to_dict(r):
        if isinstance(r, dict):
            return r
        if hasattr(r, "to_dict"):
            return r.to_dict()
        return {k: v for k, v in getattr(r, "__dict__", {}).items() if not k.startswith("_")}
    return pd.DataFrame([to_dict(r) for r in rows])

# ----------------- Main fetch -------------
all_years = []

with cfbd.ApiClient(configuration) as api_client:
    # Primary: player season stats
    stats_api = cfbd.StatsApi(api_client)

    for year in YEARS:
        print(f"Fetching player season stats for {year} ...")
        try:
            # Most recent cfbd-python exposes get_player_season_stats
            rows = stats_api.get_player_season_stats(year=year)
        except ApiException as e:
            print(f"  ⚠️ API error for {year}: {e}")
            continue
        except AttributeError:
            # Fallback for older versions
            if hasattr(stats_api, "player_season_stats"):
                rows = stats_api.player_season_stats(year=year)
            else:
                print(f"  ⚠️ No season stats method found for {year}—update cfbd package?")
                continue

        df = rows_to_df(rows)
        if df.empty:
            print(f"  (No rows for {year})")
            continue

        if "season" not in df.columns:
            df["season"] = year

        if ONLY_FANTASY_POSITIONS and "position" in df.columns:
            df = df[df["position"].isin(FANTASY_POS)].copy()

        out_year = SAVE_DIR / f"cfbd_player_season_stats_{year}.csv"
        df.to_csv(out_year, index=False)
        print(f"  Saved → {out_year} ({df.shape[0]} rows, {df.shape[1]} cols)")

        all_years.append(df)

        # be polite to the API
        time.sleep(0.25)

    # Optional: example for AdjustedMetricsApi (like your snippet)
    # Same client/auth, just a different API class:
    try:
        adj_api = cfbd.AdjustedMetricsApi(api_client)
        # Example call (remove or tweak as needed)
        adj_rows = adj_api.get_adjusted_player_passing_stats(year=2024)
        adj_df = rows_to_df(adj_rows)
        if not adj_df.empty:
            out_adj = SAVE_DIR / "cfbd_adjusted_player_passing_2024.csv"
            adj_df.to_csv(out_adj, index=False)
            print(f"  Adjusted passing saved → {out_adj} ({adj_df.shape[0]} rows)")
    except ApiException as e:
        print(f"AdjustedMetricsApi error: {e}")

# Master CSV
if all_years:
    master = pd.concat(all_years, ignore_index=True, sort=False)
    master_path = SAVE_DIR / "cfbd_player_season_stats_2014_2024_master.csv"
    master.to_csv(master_path, index=False)
    print(f"\n✅ Master saved → {master_path} ({master.shape[0]} rows, {master.shape[1]} cols)")
else:
    print("No data fetched — check token/years.")
