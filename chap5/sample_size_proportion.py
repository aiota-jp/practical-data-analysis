from scipy import stats
import numpy as np

# 例: 内閣支持率を±3%の精度で推定したい
E = 0.03  # 許容誤差（3%）
p_est = 0.5  # 最も保守的な仮定（分散最大）
z = stats.norm.ppf(0.975)

n_required = (z ** 2) * p_est * (1 - p_est) / (E ** 2)
print(f"必要サンプルサイズ: {int(np.ceil(n_required))}以上")
# → 約1068人必要

