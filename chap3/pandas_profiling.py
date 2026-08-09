# ydata-profilingのインストール
# pip install ydata-profiling

from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("profiling_data.csv")

# プロファイルレポートの生成
profile = ProfileReport(df, title="データ分析レポート", explorative=True)
profile.to_file("report.html")  # HTMLレポートとして出力
