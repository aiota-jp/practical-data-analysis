import numpy as np
import pandas as pd

data = np.array([65, 70, 75, 80, 85, 90, 95, 100, 60, 72])

# 基本的な計算
mean = np.mean(data)
print(f"平均: {mean}")

# 偏差
deviations = data - mean
print(f"偏差: {deviations}")
print(f"偏差の合計: {deviations.sum():.10f}")  # ほぼ0

# 偏差平方和
ss = np.sum(deviations**2)
print(f"偏差平方和: {ss}")

# 母分散（nで割る）
variance_pop = np.var(data, ddof=0)
print(f"母分散: {variance_pop:.2f}")

# 不偏分散（n-1で割る）- サンプルデータの場合
variance_sample = np.var(data, ddof=1)
print(f"不偏分散: {variance_sample:.2f}")

# 標準偏差
std_pop = np.std(data, ddof=0)
print(f"母標準偏差: {std_pop:.2f}")

std_sample = np.std(data, ddof=1)
print(f"標本標準偏差: {std_sample:.2f}")
