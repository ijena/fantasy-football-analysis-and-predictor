import nfl_data_py

years = range(2014,2025)

depth_charts = nfl_data_py.import_depth_charts(years)

print(depth_charts.columns)
print(depth_charts.head())