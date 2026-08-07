import pandas as pd

# CSVファイルの読み込み
df = pd.read_csv("data.csv")

# 基本情報の確認
print(df.head())       # 先頭5行を表示
print(df.describe())   # 基本統計量を表示
print(df.info())       # データ型や欠損値の情報

# データのフィルタリング
filtered = df[df["age"] >= 30]

# グループごとの集計
grouped = df.groupby("category")["sales"].mean()
