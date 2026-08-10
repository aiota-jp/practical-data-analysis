import numpy as np
import pandas as pd

# サンプルデータ（気温とビール売上本数）
temperature = np.array([22, 25, 28, 30, 32, 35, 27, 24, 33, 29])
beer_sales = np.array([150, 180, 220, 250, 280, 320, 200, 170, 300, 230])

# 共分散の計算
# 方法1: NumPyで計算
cov_matrix = np.cov(temperature, beer_sales, ddof=0)  # 母共分散
print(f"共分散行列:\n{cov_matrix}")
print(f"XとYの共分散: {cov_matrix[0, 1]:.2f}")

# 方法2: 手動計算
mean_x = np.mean(temperature)
mean_y = np.mean(beer_sales)
covariance = np.mean((temperature - mean_x) * (beer_sales - mean_y))
print(f"手動計算の共分散: {covariance:.2f}")

# 方法3: Pandasで計算
df = pd.DataFrame({"気温": temperature, "ビール売上": beer_sales})
print(f"\nPandas共分散行列:\n{df.cov()}")