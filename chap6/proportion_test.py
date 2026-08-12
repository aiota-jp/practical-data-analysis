import numpy as np
from scipy import stats

# データ
n = 100        # 試行回数（購入者数）
k = 12         # 成功回数（キャッシュバック対象者数）
p_0 = 0.20     # 帰無仮説の比率

# 標本比率
p_hat = k / n

print(f"=== 比率の検定 ===")
print(f"標本サイズ n: {n}")
print(f"キャッシュバック対象者: {k}名")
print(f"標本比率 p̂: {p_hat:.2f} ({p_hat*100:.1f}%)")
print(f"帰無仮説の比率 p₀: {p_0:.2f} ({p_0*100:.1f}%)")

# 方法1: 正規近似による検定
z = (p_hat - p_0) / np.sqrt(p_0 * (1 - p_0) / n)
p_value_norm = 2 * stats.norm.cdf(-abs(z))  # 両側検定

print(f"\n=== 正規近似による検定 ===")
print(f"Z値: {z:.4f}")
print(f"p値: {p_value_norm:.6f}")

# 方法2: scipy.stats.binomtest（正確な二項検定）
result = stats.binomtest(k, n, p_0, alternative='two-sided')

print(f"\n=== 二項検定（正確法） ===")
print(f"p値: {result.pvalue:.6f}")

# 判定
alpha = 0.05
print(f"\n=== 判定（有意水準 α = {alpha}） ===")
if result.pvalue < alpha:
    print(f"p値({result.pvalue:.6f}) < α({alpha})")
    print("→ 帰無仮説を棄却。")
    print("  「購入者の20%がキャッシュバックを受ける」という主張は統計的に否定される。")
else:
    print(f"p値({result.pvalue:.6f}) ≥ α({alpha})")
    print("→ 帰無仮説を棄却できない。偶然の範囲内と判断される。")