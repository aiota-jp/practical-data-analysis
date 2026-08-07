import pandas as pd
import numpy as np

# サンプルデータの作成
df = pd.DataFrame({
    "name": ["田中", "鈴木", None, "佐藤", "山田"],
    "age": [25, np.nan, 30, 28, np.nan],
    "salary": [300000, 450000, np.nan, 350000, 280000],
    "department": ["営業", "開発", "営業", None, "開発"]
})

# 元のデータを表示
print("===== 元のデータ =====")
print(df)


# ========================================
# 1. 固定値で補完
# ========================================

df_filled = df.copy()

df_filled["age"] = df_filled["age"].fillna(0)

print("\n===== 固定値（0）で補完 =====")
print(df_filled)


# ========================================
# 2. 平均値で補完
# ========================================

df_filled = df.copy()

mean_age = df_filled["age"].mean()
df_filled["age"] = df_filled["age"].fillna(mean_age)

print("\n===== 平均値で補完 =====")
print("ageの平均値：", mean_age)
print(df_filled)


# ========================================
# 3. 中央値で補完
# ========================================

df_filled = df.copy()

median_salary = df_filled["salary"].median()
df_filled["salary"] = df_filled["salary"].fillna(median_salary)

print("\n===== 中央値で補完 =====")
print("salaryの中央値：", median_salary)
print(df_filled)


# ========================================
# 4. 最頻値で補完
# ========================================

df_filled = df.copy()

mode_department = df_filled["department"].mode()[0]
df_filled["department"] = df_filled["department"].fillna(mode_department)

print("\n===== 最頻値で補完 =====")
print("departmentの最頻値：", mode_department)
print(df_filled)


# ========================================
# 5. 前の値で補完（前方補完）
# ========================================

df_filled = df.copy()

df_filled["age"] = df_filled["age"].ffill()

print("\n===== 前の値で補完（前方補完） =====")
print(df_filled)


# ========================================
# 6. 後ろの値で補完（後方補完）
# ========================================

df_filled = df.copy()

df_filled["age"] = df_filled["age"].bfill()

print("\n===== 後ろの値で補完（後方補完） =====")
print(df_filled)


# ========================================
# 7. 線形補間
# ========================================

df_filled = df.copy()

df_filled["age"] = df_filled["age"].interpolate(method="linear")

print("\n===== 線形補間 =====")
print(df_filled)