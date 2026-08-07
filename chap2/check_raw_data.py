import pandas as pd

# 前処理前のデータを読み込む
df = pd.read_csv("sample_data.csv")

# 基本情報の確認
print("===== データサイズ =====")
print(df.shape)

print("\n===== データ型 =====")
print(df.dtypes)

print("\n===== 基本情報 =====")
df.info()

print("\n===== 基本統計量 =====")
print(df.describe())

print("\n===== 先頭10行 =====")
print(df.head(10))