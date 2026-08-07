import pandas as pd

# 基本的なCSV読み込み
df = pd.read_csv("data.csv")

# エンコーディングを指定（日本語を含む場合）
df = pd.read_csv("data.csv", encoding="utf-8")
# df = pd.read_csv("data.csv", encoding="shift_jis")
print(df)

# ヘッダーがない場合
df = pd.read_csv("data.csv", header=None, names=["col1", "col2", "col3"])
print(df)

# 特定の列だけ読み込む
df = pd.read_csv("data.csv", usecols=["name", "age", "salary"])
print(df)

# インデックス列を指定
df = pd.read_csv("data.csv", index_col="id")
print(df)
