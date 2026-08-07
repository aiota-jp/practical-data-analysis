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
# 1. 欠損値を含む行を削除
# ========================================

df_dropped = df.dropna()

print("\n===== 欠損値を含む行を削除 =====")
print(df_dropped)


# ========================================
# 2. 特定の列に欠損がある行のみ削除
# ========================================

df_dropped = df.dropna(subset=["age", "salary"])

print("\n===== age または salary に欠損がある行を削除 =====")
print(df_dropped)


# ========================================
# 3. 全ての値が欠損の行を削除
# ========================================

df_dropped = df.dropna(how="all")

print("\n===== 全ての値が欠損している行を削除 =====")
print(df_dropped)


# ========================================
# 4. 欠損値が一定割合以上の列を削除
# ========================================

threshold = len(df) * 0.3

df_dropped = df.dropna(axis=1, thresh=threshold)

print("\n===== 一定数以上のデータが存在する列を残す =====")
print("しきい値:", threshold)
print(df_dropped)