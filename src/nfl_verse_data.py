import nfl_data_py


def load_depth_chart_data(years):


    depth_charts = nfl_data_py.import_depth_charts(years)

    #filter for fantasy relevant positions
    fantasy_positions = ["QB", "RB", "TE", "WR"]
    return depth_charts[depth_charts["position"].isin(fantasy_positions)].copy()

years = range(2014,2025)
depth_chart = load_depth_chart_data(years)

