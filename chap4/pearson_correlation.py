import numpy as np
import pandas as pd

# サンプルデータ（気温とビール売上）
temperature = np.array([22, 25, 28, 30, 32, 35, 27, 24, 33, 29])
beer_sales = np.array([150, 180, 220, 250, 280, 320, 200, 170, 300, 230])

# 方法1: NumPyで計算
corr_matrix = np.corrcoef(temperature, beer_sales)
print(f"相関係数: {corr_matrix[0, 1]:.4f}")

# 方法2: Pandasで計算
df = pd.DataFrame({"気温": temperature, "ビール売上": beer_sales})
print(f"\n相関行列:\n{df.corr()}")

# 方法3: SciPyで計算（p値付き）
from scipy import stats
r, p_value = stats.pearsonr(temperature, beer_sales)
print(f"\n相関係数 R: {r:.4f}")
print(f"p値: {p_value:.6f}")

if p_value < 0.05:
    print("→ 統計的に有意な相関あり（p < 0.05）")
else:
    print("→ 統計的に有意な相関なし")