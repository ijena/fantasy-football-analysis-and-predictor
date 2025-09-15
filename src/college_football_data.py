import os
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

import cfbd
from cfbd.rest import ApiException

# ----------------- Config -----------------
YEARS = range(2014, 2025)  # 2014–2024 inclusive
SAVE_DIR = Path(r"C:\Users\idhan\Downloads\Nerds with Numbers\fantasy-football-analysis-and-predictor\data\college_football_data 2014-2024")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FANTASY_POS = {"QB", "RB", "WR", "TE"}  # optional filter
ONLY_FANTASY_POSITIONS = True           # set False to keep all positions

# ----------------- Auth -------------------
load_dotenv()
token = os.getenv("BEARER_TOKEN")  # in .env: BEARER_TOKEN=your_api_key (no quotes)
if not token:
    raise RuntimeError("BEARER_TOKEN missing. Put it in your .env as BEARER_TOKEN=... (no quotes)")

# Configure cfbd to use Bearer token
configuration = cfbd.Configuration(
    host="https://api.collegefootballdata.com",
    access_token=token   # cfbd-python will send Authorization: Bearer <token>
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

def save_year_df(df: pd.DataFrame, year: int, stem: str):
    out_path = SAVE_DIR / f"{stem}_{year}.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path} ({df.shape[0]} rows, {df.shape[1]} cols)")

# ----------------- Main fetch -------------
# // Use one API client context
with cfbd.ApiClient(configuration) as api_client:
    stats_api = cfbd.StatsApi(api_client)

    all_years = []

    for year in YEARS:
        print(f"Fetching player season stats for {year} ...")
        try:
            # Preferred method name in current cfbd-python
            rows = stats_api.get_player_season_stats(year=year)
        except AttributeError:
            # Older version fallback
            if hasattr(stats_api, "player_season_stats"):
                rows = stats_api.player_season_stats(year=year)
            else:
                print(f"  ⚠️ No season stats method found for {year}—update cfbd package?")
                continue
        except ApiException as e:
            # Helpful guidance on 401
            if e.status == 401:
                print("  ❌ Unauthorized (401). Make sure BEARER_TOKEN is valid and sent as Bearer.")
            else:
                print(f"  ⚠️ API error for {year}: {e}")
            continue

        df = rows_to_df(rows)
        if df.empty:
            print(f"  (No rows for {year})")
            continue

        # Normalize
        if "season" not in df.columns:
            df["season"] = year

        # Optional fantasy positions filter
        if ONLY_FANTASY_POSITIONS and "position" in df.columns:
            df = df[df["position"].isin(FANTASY_POS)].copy()

        # Save per-year
        save_year_df(df, year, stem="cfbd_player_season_stats")
        all_years.append(df)

        # Be polite to API
        time.sleep(0.25)

    # ----- OPTIONAL: Adjusted metrics examples (per your request) -----
    # Uncomment any of these blocks if you want adjusted stats, too.

    # adj_api = cfbd.AdjustedMetricsApi(api_client)

    # # Adjusted PASSING (QBs)
    # for year in YEARS:
    #     try:
    #         adj_rows = adj_api.get_adjusted_player_passing_stats(year=year)
    #         adj_df = rows_to_df(adj_rows)
    #         if not adj_df.empty:
    #             save_year_df(adj_df, year, stem="cfbd_adjusted_player_passing")
    #             time.sleep(0.25)
    #     except ApiException as e:
    #         print(f"Adjusted passing error {year}: {e}")

    # # Adjusted RECEIVING (WR/TE/RB)
    # for year in YEARS:
    #     try:
    #         adj_rows = adj_api.get_adjusted_player_receiving_stats(year=year)
    #         adj_df = rows_to_df(adj_rows)
    #         if not adj_df.empty:
    #             save_year_df(adj_df, year, stem="cfbd_adjusted_player_receiving")
    #             time.sleep(0.25)
    #     except ApiException as e:
    #         print(f"Adjusted receiving error {year}: {e}")

    # # Adjusted RUSHING (RB/QB)
    # for year in YEARS:
    #     try:
    #         adj_rows = adj_api.get_adjusted_player_rushing_stats(year=year)
    #         adj_df = rows_to_df(adj_rows)
    #         if not adj_df.empty:
    #             save_year_df(adj_df, year, stem="cfbd_adjusted_player_rushing")
    #             time.sleep(0.25)
    #     except ApiException as e:
    #         print(f"Adjusted rushing error {year}: {e}")

# Build a master CSV for the season stats
if all_years:
    master = pd.concat(all_years, ignore_index=True, sort=False)
    master_path = SAVE_DIR / "cfbd_player_season_stats_2014_2024_master.csv"
    master.to_csv(master_path, index=False)
    print(f"\n✅ Master saved → {master_path} ({master.shape[0]} rows, {master.shape[1]} cols)")
else:
    print("No data fetched — check token/years.")
