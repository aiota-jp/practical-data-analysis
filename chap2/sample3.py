import pandas as pd

# ========================================
# チャンク（分割）で読み込み
# ========================================
print("===== チャンク（分割）で読み込み =====")
chunk_no = 1
for chunk in pd.read_csv("large_data.csv", chunksize=10000):
    print(f"\n--- チャンク {chunk_no} ---")
    print(f"データ件数：{len(chunk)}件")
    print(chunk)
    chunk_no += 1

# ========================================
# データ型を指定してメモリ削減
# ========================================
print("\n===== データ型を指定して読み込み =====")
df = pd.read_csv("data.csv", dtype={
    "id": "int32",
    "category": "category",
    "value": "float32"
})
print(df)

print("\n===== データ型 =====")
print(df.dtypes)