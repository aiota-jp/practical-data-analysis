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

# 月ごとの平均を計算
monthly_pattern = ts.groupby(ts.index.month).mean()

# 月別平均売上の可視化
plt.figure(figsize=(8, 5))
plt.bar(monthly_pattern.index, monthly_pattern.values, color="steelblue", alpha=0.8)
plt.xlabel("月")
plt.ylabel("平均売上")
plt.title("月別平均売上（季節パターン）")
plt.xticks(range(1, 13))
plt.grid(axis="y", alpha=0.3)
plt.show()