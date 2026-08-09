import numpy as np
from scipy import stats

# テストスコアの標準化
scores = np.array([55, 60, 65, 70, 75, 80, 85, 90, 95])
mean = np.mean(scores)
std = np.std(scores)

# 標準化（Zスコアの計算）
z_scores = (scores - mean) / std
print(f"元のデータ: {scores}")
print(f"平均: {mean}, 標準偏差: {std:.2f}")
print(f"Zスコア: {np.round(z_scores, 2)}")

# 特定のスコアの偏差値を計算
# 偏差値 = 50 + 10 × Zスコア
deviation_values = 50 + 10 * z_scores
print(f"偏差値: {np.round(deviation_values, 1)}")
