import numpy as np
from scipy import stats

# 例: ある工場の製品重量（母標準偏差σ=5gが既知）
sample = np.array([198, 202, 195, 201, 199, 203, 197, 200, 204, 196,
                   201, 198, 203, 199, 200, 202, 197, 201, 199, 200,
                   198, 203, 201, 199, 200, 202, 197, 201, 199, 200])
sigma = 5  # 母標準偏差（既知）
n = len(sample)
x_bar = sample.mean()

# 95%信頼区間（正規分布）
confidence = 0.95
z = stats.norm.ppf((1 + confidence) / 2)  # 1.96
se = sigma / np.sqrt(n)

lower = x_bar - z * se
upper = x_bar + z * se

print(f"=== 正規分布による母平均の区間推定 ===")
print(f"標本平均: {x_bar:.2f} g")
print(f"標準誤差 SE: {se:.4f}")
print(f"z値 (95%): {z:.4f}")
print(f"95%信頼区間: [{lower:.2f}, {upper:.2f}] g")

# scipy.stats.norm.interval を使う方法
ci = stats.norm.interval(confidence, loc=x_bar, scale=se)
print(f"scipy利用: [{ci[0]:.2f}, {ci[1]:.2f}] g")