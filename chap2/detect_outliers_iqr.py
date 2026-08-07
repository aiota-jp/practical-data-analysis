import pandas as pd

# サンプルデータの作成
df = pd.DataFrame({
    "value": [10, 12, 14, 15, 13, 11, 100, 14, 12, 13, -20]
})

print("===== 元のデータ =====")
print(df)


# ========================================
# IQR法による外れ値の検出
# ========================================

Q1 = df["value"].quantile(0.25)
Q3 = df["value"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\n===== IQR =====")
print("第1四分位数（Q1）:", Q1)
print("第3四分位数（Q3）:", Q3)
print("IQR:", IQR)

print("\n===== 外れ値の判定範囲 =====")
print("下限:", lower_bound)
print("上限:", upper_bound)


# 外れ値を検出
outliers = df[
    (df["value"] < lower_bound) |
    (df["value"] > upper_bound)
]

print("\n===== 外れ値 =====")
print(outliers)


# 外れ値を除外
df_clean = df[
    (df["value"] >= lower_bound) &
    (df["value"] <= upper_bound)
]

print("\n===== 外れ値を除外したデータ =====")
print(df_clean)