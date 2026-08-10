import numpy as np
from scipy import stats

# 例: 新薬の効果測定（15人の被験者）
sample = np.array([5.2, 4.8, 6.1, 5.5, 4.9, 5.7, 6.3, 5.0,
                   5.4, 4.6, 5.8, 6.0, 5.3, 4.7, 5.6])
n = len(sample)
x_bar = sample.mean()
s = sample.std(ddof=1)  # 不偏標準偏差
se = s / np.sqrt(n)

# 95%信頼区間（t分布）
confidence = 0.95
df = n - 1  # 自由度
t_value = stats.t.ppf((1 + confidence) / 2, df)

lower = x_bar - t_value * se
upper = x_bar + t_value * se

print(f"=== t分布による母平均の区間推定 ===")
print(f"標本サイズ n: {n}")
print(f"標本平均: {x_bar:.4f}")
print(f"不偏標準偏差 s: {s:.4f}")
print(f"標準誤差 SE: {se:.4f}")
print(f"自由度: {df}")
print(f"t値 (95%, df={df}): {t_value:.4f}")
print(f"95%信頼区間: [{lower:.4f}, {upper:.4f}]")

# scipy.stats.t.interval を使う方法
ci = stats.t.interval(confidence, df, loc=x_bar, scale=se)
print(f"scipy利用: [{ci[0]:.4f}, {ci[1]:.4f}]")