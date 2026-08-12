import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
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

# 1次差分（トレンド除去）
ts_diff1 = ts.diff().dropna()

# 季節差分（季節性除去）
ts_seasonal_diff = ts.diff(12).dropna()

# 可視化
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
axes[0].plot(ts)
axes[0].set_title("元データ（非定常）")
axes[0].grid(alpha=0.3)

axes[1].plot(ts_diff1)
axes[1].set_title("1次差分（トレンド除去）")
axes[1].grid(alpha=0.3)

axes[2].plot(ts_seasonal_diff)
axes[2].set_title("季節差分（周期12）")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 1次差分後のADF検定
result = adfuller(ts_diff1)
print(f"1次差分のADF検定 p値: {result[1]:.6f}")

if result[1] < 0.05:
    print("→ 1次差分後のデータは定常と判断できます。")
else:
    print("→ 1次差分後も非定常の可能性があります。")