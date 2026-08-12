import numpy as np
from scipy import stats

# 牛肉30パックの重量データ（g）
weights = np.array([97, 99, 96, 98, 95, 101, 97, 94, 98, 96,
                    99, 97, 95, 98, 96, 100, 97, 94, 99, 96,
                    98, 97, 95, 99, 96, 98, 97, 94, 100, 96])

# 基本統計量
n = len(weights)
x_bar = weights.mean()
s = weights.std(ddof=1)
se = s / np.sqrt(n)

print(f"=== 基本統計量 ===")
print(f"標本サイズ n: {n}")
print(f"標本平均: {x_bar:.2f} g")
print(f"不偏標準偏差: {s:.2f} g")
print(f"標準誤差 SE: {se:.4f}")

# 仮説設定
mu_0 = 100  # 帰無仮説: μ = 100
print(f"\n=== 仮説 ===")
print(f"H₀: μ = {mu_0}g（牛肉1パックの平均重量は100gである）")
print(f"H₁: μ ≠ {mu_0}g（牛肉1パックの平均重量は100gではない）")

# t値の手動計算
t_manual = (x_bar - mu_0) / se
print(f"\n=== t値の計算 ===")
print(f"t = (x̄ - μ₀) / SE = ({x_bar:.2f} - {mu_0}) / {se:.4f} = {t_manual:.4f}")

# scipy.stats.ttest_1samp による1標本t検定
t_stat, p_value = stats.ttest_1samp(weights, mu_0)

print(f"\n=== 検定結果 ===")
print(f"t値: {t_stat:.4f}")
print(f"p値: {p_value:.6f}")

# 判定（有意水準 5%）
alpha = 0.05
print(f"\n=== 判定（有意水準 α = {alpha}） ===")
if p_value < alpha:
    print(f"p値({p_value:.6f}) < α({alpha})")
    print("→ 帰無仮説を棄却。牛肉1パックの平均重量は100gとは言えない。")
else:
    print(f"p値({p_value:.6f}) ≥ α({alpha})")
    print("→ 帰無仮説を棄却できない。100gでないとは断定できない。")