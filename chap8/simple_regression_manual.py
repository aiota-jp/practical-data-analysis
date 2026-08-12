import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import japanize_matplotlib

# サンプルデータ（気温とビール売上）
np.random.seed(42)
temperature = np.array([18, 20, 22, 24, 25, 26, 27, 28, 30, 32, 33, 35])
beer_sales = 50 + 8 * temperature + np.random.normal(0, 15, len(temperature))

# 手動で最小二乗法を計算
x_mean = temperature.mean()
y_mean = beer_sales.mean()

# 傾き a
numerator = np.sum((temperature - x_mean) * (beer_sales - y_mean))
denominator = np.sum((temperature - x_mean) ** 2)
a = numerator / denominator

# 切片 b
b = y_mean - a * x_mean

print(f"=== 最小二乗法による単回帰分析 ===")
print(f"傾き a: {a:.4f}")
print(f"切片 b: {b:.4f}")
print(f"回帰式: y = {a:.2f}x + {b:.2f}")
print(f"\n解釈: 気温が1℃上がると、ビール売上は約{a:.1f}本増加する")