import pandas as pd

# Excelファイル
print("===== Excelファイルの読み込み =====")
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
print(df)

# TSV（タブ区切り）
print("\n===== TSVファイルの読み込み =====")
df = pd.read_csv("data.tsv", sep="\t")
print(df)

# JSON
print("\n===== JSONファイルの読み込み =====")
df = pd.read_json("data.json")
print(df)

# クリップボード（Excelからコピーしたデータ）
print("\n===== クリップボードからの読み込み =====")
df = pd.read_clipboard()
print(df)