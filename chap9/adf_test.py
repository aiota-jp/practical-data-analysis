import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# サンプル時系列データの生成（月次売上データ）
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", periods=72, freq="MS")
trend = np.linspace(100, 200, 72)
seasonal = 30 * np.sin(2 * np.pi * np.arange(72) / 12)
noise = np.random.normal(0, 10, 72)
sales = trend + seasonal + noise

# 時系列データの作成
ts = pd.Series(sales, index=dates, name="月次売上")

# ADF検定
result = adfuller(ts)

print("=== ADF検定（定常性の検定） ===")
print(f"ADF統計量: {result[0]:.4f}")
print(f"p値: {result[1]:.6f}")
print(f"ラグ数: {result[2]}")

if result[1] < 0.05:
    print("→ 定常である（帰無仮説「単位根あり」を棄却）")
else:
    print("→ 非定常である（差分を取って定常化を検討）")