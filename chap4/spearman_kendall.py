import numpy as np
import pandas as pd
from scipy import stats

# サンプルデータ（気温とビール売上）
temperature = np.array([22, 25, 28, 30, 32, 35, 27, 24, 33, 29])
beer_sales = np.array([150, 180, 220, 250, 280, 320, 200, 170, 300, 230])

# スピアマンの順位相関係数
rho, p_value = stats.spearmanr(temperature, beer_sales)
print(f"スピアマンの順位相関係数: {rho:.4f}")
print(f"p値: {p_value:.6f}")

# ケンドールの順位相関係数
tau, p_value = stats.kendalltau(temperature, beer_sales)
print(f"ケンドールの順位相関係数: {tau:.4f}")