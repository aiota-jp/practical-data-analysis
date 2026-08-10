import numpy as np
import pandas as pd

# 母集団の生成（例: 全社員10000人の年収データ）
np.random.seed(42)
population = np.random.normal(loc=500, scale=100, size=10000)  # 平均500万円、標準偏差100万円

print(f"母集団のサイズ: {len(population)}")
print(f"母平均 μ: {population.mean():.2f}万円")
print(f"母標準偏差 σ: {population.std():.2f}万円")

# 標本の抽出（単純無作為抽出）
sample_size = 100
sample = np.random.choice(population, size=sample_size, replace=False)

print(f"\n標本のサイズ: {len(sample)}")
print(f"標本平均: {sample.mean():.2f}万円")
print(f"標本標準偏差: {sample.std(ddof=1):.2f}万円")