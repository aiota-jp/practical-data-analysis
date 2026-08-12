import numpy as np
from sklearn.linear_model import LinearRegression

# サンプルデータ（気温とビール売上）
np.random.seed(42)
temperature = np.array([18, 20, 22, 24, 25, 26, 27, 28, 30, 32, 33, 35])
beer_sales = 50 + 8 * temperature + np.random.normal(0, 15, len(temperature))

# モデルの構築
X = temperature.reshape(-1, 1)  # 2次元配列に変換
y = beer_sales

lr = LinearRegression()
lr.fit(X, y)

# 結果表示
print("=== scikit-learn による単回帰分析 ===")
print(f"傾き: {lr.coef_[0]:.4f}")
print(f"切片: {lr.intercept_:.4f}")
print(f"R²スコア: {lr.score(X, y):.4f}")

# 予測
new_temp = np.array([[36]])  # 気温36℃のときの売上を予測
predicted = lr.predict(new_temp)
print(f"\n気温36℃のときの予測売上: {predicted[0]:.1f}本")