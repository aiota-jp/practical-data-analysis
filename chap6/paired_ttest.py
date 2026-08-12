import numpy as np
from scipy import stats

# 例: 文系の生徒20名の前期・後期の数学テスト
np.random.seed(42)
before = np.array([55, 62, 48, 70, 58, 65, 52, 60, 45, 68,
                   57, 63, 50, 72, 55, 61, 49, 66, 53, 59])
after = np.array([60, 68, 55, 73, 65, 70, 58, 67, 52, 72,
                  62, 69, 56, 75, 60, 66, 55, 71, 58, 65])

# 差の計算
diff = after - before
print(f"=== 基本統計量 ===")
print(f"前期: 平均={before.mean():.2f}, 標準偏差={before.std(ddof=1):.2f}")
print(f"後期: 平均={after.mean():.2f}, 標準偏差={after.std(ddof=1):.2f}")
print(f"差(後期-前期): 平均={diff.mean():.2f}, 標準偏差={diff.std(ddof=1):.2f}")

# 仮説
print(f"\n=== 仮説 ===")
print("H₀: 前期と後期で数学の学力に差はない（μ_diff = 0）")
print("H₁: 前期と後期で数学の学力に差がある（μ_diff ≠ 0）")

# 対応のあるt検定
t_stat, p_value = stats.ttest_rel(before, after)

print(f"\n=== 対応のあるt検定の結果 ===")
print(f"t値: {t_stat:.4f}")
print(f"p値: {p_value:.8f}")

alpha = 0.05
if p_value < alpha:
    print(f"\n→ p値({p_value:.6f}) < α({alpha}): 帰無仮説を棄却")
    print("  前期と後期で数学の成績に統計的に有意な差がある。")
    if diff.mean() > 0:
        print(f"  後期の方が平均{diff.mean():.1f}点高い。")
else:
    print(f"\n→ p値({p_value:.6f}) ≥ α({alpha}): 帰無仮説を棄却できない")