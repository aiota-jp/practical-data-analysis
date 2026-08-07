import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# サンプルデータの作成
df = pd.DataFrame({
    "age": [25, 30, 35, 40, 45],
    "salary": [3000000, 4500000, 5000000, 7000000, 8000000]
})

print("===== 変換前 =====")
print(df)


# ========================================
# Min-Max法による正規化
# ========================================

scaler = MinMaxScaler()

df_normalized = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)

print("\n===== 正規化後 =====")
print(df_normalized)