import nfl_data_py

years = range(2014,2025)

depth_charts = nfl_data_py.import_depth_charts(years)
print(sorted(depth_charts["position"].dropna().unique()))

#filter for fantasy relevant positions
fantasy_positions = ["QB", "RB", "TE", "WR"]
depth_charts = depth_charts[depth_charts["position"].isin(fantasy_positions)].copy()


print(sorted(depth_charts["position"].dropna().unique()))
