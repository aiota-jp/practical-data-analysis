import pandas as pd
import numpy as np

# サンプルデータの作成
np.random.seed(42)
df = pd.DataFrame({
    "age": [25, 30, 35, 28, 42, 38, 45, 33, 29, 50],
    "salary": [300, 450, 500, 380, 700, 600, 800, 480, 350, 900],
    "score": [72, 85, 90, 68, 95, 88, 92, 78, 80, 98]
})

# 各統計量を個別に算出
print(f"合計: {df['salary'].sum()}")
print(f"平均値: {df['salary'].mean()}")
print(f"中央値: {df['salary'].median()}")
print(f"最頻値: {df['salary'].mode()[0]}")
print(f"最大値: {df['salary'].max()}")
print(f"最小値: {df['salary'].min()}")
print(f"範囲: {df['salary'].max() - df['salary'].min()}")
print(f"分散: {df['salary'].var()}")
print(f"標準偏差: {df['salary'].std()}")
print(f"四分位範囲(IQR): {df['salary'].quantile(0.75) - df['salary'].quantile(0.25)}")

