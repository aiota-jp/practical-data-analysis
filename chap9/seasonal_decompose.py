import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from statsmodels.tsa.seasonal import seasonal_decompose

# サンプル時系列データの生成（月次売上データ）
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", periods=72, freq="MS")
trend = np.linspace(100, 200, 72)
seasonal = 30 * np.sin(2 * np.pi * np.arange(72) / 12)
noise = np.random.normal(0, 10, 72)
sales = trend + seasonal + noise

# 時系列データの作成
ts = pd.Series(sales, index=dates, name="月次売上")

# 季節分解（加法モデル）
decomposition = seasonal_decompose(ts, model="additive", period=12)

# 可視化
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
decomposition.observed.plot(ax=axes[0], title="元データ（Observed）")
decomposition.trend.plot(ax=axes[1], title="トレンド（Trend）")
decomposition.seasonal.plot(ax=axes[2], title="季節変動（Seasonal）")
decomposition.resid.plot(ax=axes[3], title="残差（Residual）")

for ax in axes:
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()