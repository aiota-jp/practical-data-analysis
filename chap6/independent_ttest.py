import numpy as np
from scipy import stats

# 例: 理系と文系の数学テスト成績
np.random.seed(42)
science_scores = np.array([78, 85, 92, 88, 76, 90, 84, 95, 82, 87,
                           91, 79, 86, 93, 80, 88, 84, 90, 77, 85])
liberal_arts_scores = np.array([65, 72, 58, 70, 68, 75, 62, 71, 66, 73,
                                60, 69, 74, 63, 67, 70, 64, 72, 61, 68])

print(f"=== 基本統計量 ===")
print(f"理系: 平均={science_scores.mean():.2f}, 標準偏差={science_scores.std(ddof=1):.2f}, n={len(science_scores)}")
print(f"文系: 平均={liberal_arts_scores.mean():.2f}, 標準偏差={liberal_arts_scores.std(ddof=1):.2f}, n={len(liberal_arts_scores)}")

# 仮説
print(f"\n=== 仮説 ===")
print("H₀: 理系と文系の数学の平均点に差はない（μ₁ = μ₂）")
print("H₁: 理系と文系の数学の平均点に差がある（μ₁ ≠ μ₂）")

# 等分散性の検定（Levene検定）
levene_stat, levene_p = stats.levene(science_scores, liberal_arts_scores)
print(f"\n=== 等分散性の検定（Levene検定） ===")
print(f"統計量: {levene_stat:.4f}, p値: {levene_p:.4f}")

if levene_p >= 0.05:
    print("→ 等分散を仮定できる（Studentのt検定を使用）")
    equal_var = True
else:
    print("→ 等分散を仮定できない（Welchのt検定を使用）")
    equal_var = False

# 独立t検定
t_stat, p_value = stats.ttest_ind(science_scores, liberal_arts_scores, equal_var=equal_var)

print(f"\n=== 独立t検定の結果 ===")
print(f"t値: {t_stat:.4f}")
print(f"p値: {p_value:.8f}")

alpha = 0.05
if p_value < alpha:
    print(f"\n→ p値({p_value:.6f}) < α({alpha}): 帰無仮説を棄却")
    print("  理系と文系の数学の平均点には統計的に有意な差がある。")
else:
    print(f"\n→ p値({p_value:.6f}) ≥ α({alpha}): 帰無仮説を棄却できない")