import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# サンプル時系列データの生成（月次売上データ）
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", periods=72, freq="MS")
trend = np.linspace(100, 200, 72)
seasonal = 30 * np.sin(2 * np.pi * np.arange(72) / 12)
noise = np.random.normal(0, 10, 72)
sales = trend + seasonal + noise

# 時系列データの作成
ts = pd.Series(sales, index=dates, name="月次売上")

# 12ヶ月移動平均
ts_ma12 = ts.rolling(window=12, center=True).mean()

# 可視化
plt.figure(figsize=(12, 5))
plt.plot(ts, alpha=0.7, label="元データ", color="steelblue")
plt.plot(ts_ma12, linewidth=2.5, label="12ヶ月移動平均（トレンド）", color="red")
plt.xlabel("日付")
plt.ylabel("売上")
plt.title("移動平均によるトレンドの抽出")
plt.legend()
plt.grid(alpha=0.3)
plt.show()