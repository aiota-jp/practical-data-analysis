from scipy import stats
import numpy as np

# さまざまな分布のデータ
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)        # 正規分布
right_skewed = np.random.exponential(2, 1000)      # 右に歪んだ分布
left_skewed = -np.random.exponential(2, 1000)      # 左に歪んだ分布

# 歪度の計算
print(f"正規分布の歪度: {stats.skew(normal_data):.4f}")
print(f"右に歪んだ分布の歪度: {stats.skew(right_skewed):.4f}")
print(f"左に歪んだ分布の歪度: {stats.skew(left_skewed):.4f}")

# 尖度の計算
print(f"\n正規分布の尖度: {stats.kurtosis(normal_data):.4f}")
print(f"右に歪んだ分布の尖度: {stats.kurtosis(right_skewed):.4f}")
