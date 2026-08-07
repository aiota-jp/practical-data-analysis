import pandas as pd

# CSVファイルの読み込み
df = pd.read_csv("data.csv")

# 基本情報の確認
print("===== 先頭5行 =====")
print(df.head())

print("\n===== 基本統計量 =====")
print(df.describe())

print("\n===== データ型・欠損値情報 =====")
print(df.info())

# データのフィルタリング
print("\n===== 年齢30歳以上のデータ =====")
filtered = df[df["age"] >= 30]
print(filtered)

# グループごとの集計
print("\n===== カテゴリごとの平均売上 =====")
grouped = df.groupby("category")["sales"].mean()
print(grouped)