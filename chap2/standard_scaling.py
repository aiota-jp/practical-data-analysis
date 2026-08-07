import pandas as pd
from sklearn.preprocessing import StandardScaler

# サンプルデータの作成
df = pd.DataFrame({
    "age": [25, 30, 35, 40, 45],
    "salary": [3000000, 4500000, 5000000, 7000000, 8000000]
})

print("===== 変換前 =====")
print(df)


# ========================================
# Zスコア標準化
# ========================================

scaler = StandardScaler()

df_standardized = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)

print("\n===== 標準化後 =====")
print(df_standardized)

print("\n===== 標準化後の平均 =====")
print(df_standardized.mean())

print("\n===== 標準化後の標準偏差 =====")
print(df_standardized.std(ddof=0))