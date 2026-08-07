import pandas as pd
import numpy as np

# サンプルデータ
df = pd.DataFrame({
    "age": [25, 30, np.nan, 40, 35],
    "date": [
        "2024/01/01",
        "2024/02/15",
        "2024/03/20",
        "2024/04/10",
        "2024/05/05"
    ]
})

print("===== 変換前 =====")
print(df)

print("\n===== 変換前のデータ型 =====")
print(df.dtypes)


# ========================================
# NaNを含む数値 → 整数型
# ========================================

# 通常のintではなく
# pandasのNullable Integer型を使用
df["age"] = df["age"].astype("Int64")


# ========================================
# 日付フォーマットを指定して変換
# ========================================

df["date"] = pd.to_datetime(
    df["date"],
    format="%Y/%m/%d"
)


print("\n===== 変換後 =====")
print(df)

print("\n===== 変換後のデータ型 =====")
print(df.dtypes)