import pandas as pd
import numpy as np

# サンプルデータの作成
df = pd.DataFrame({
    "value": [10, 12, 14, 15, 13, 11, 100, 14, 12, 13, -20]
})

print("===== 元のデータ =====")
print(df)


# ========================================
# IQRから外れ値の範囲を求める
# ========================================

Q1 = df["value"].quantile(0.25)
Q3 = df["value"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\n===== 外れ値の判定範囲 =====")
print("下限:", lower_bound)
print("上限:", upper_bound)


# ========================================
# 方法1：外れ値を含む行を削除
# ========================================

df_clean = df[
    (df["value"] >= lower_bound) &
    (df["value"] <= upper_bound)
]

print("\n===== 方法1：外れ値を削除 =====")
print(df_clean)


# ========================================
# 方法2：上限・下限値で置き換え
# ========================================

df_clip = df.copy()

df_clip["value"] = df_clip["value"].clip(
    lower=lower_bound,
    upper=upper_bound
)

print("\n===== 方法2：クリッピング =====")
print(df_clip)


# ========================================
# 方法3：中央値で置き換え
# ========================================

df_median = df.copy()

median_val = df_median["value"].median()

condition = (
    (df_median["value"] < lower_bound) |
    (df_median["value"] > upper_bound)
)

df_median.loc[condition, "value"] = median_val

print("\n===== 方法3：中央値で置き換え =====")
print("中央値:", median_val)
print(df_median)