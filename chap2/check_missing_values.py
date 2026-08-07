import pandas as pd
import numpy as np

# サンプルデータの作成
df = pd.DataFrame({
    "name": ["田中", "鈴木", None, "佐藤", "山田"],
    "age": [25, np.nan, 30, 28, np.nan],
    "salary": [300000, 450000, np.nan, 350000, 280000],
    "department": ["営業", "開発", "営業", None, "開発"]
})

# 欠損値の確認
print(df.isnull())            # 各セルがNaNかどうか（True/False）
print(df.isnull().sum())      # 列ごとの欠損値の数
print(df.isnull().sum().sum())  # 全体の欠損値の合計

# 欠損値の割合を確認
print(df.isnull().mean() * 100)  # 列ごとの欠損割合（%）