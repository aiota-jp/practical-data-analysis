import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats

# データの準備
np.random.seed(42)
lecture = np.random.normal(loc=65, scale=10, size=30)       # 通常講義
online = np.random.normal(loc=70, scale=12, size=30)        # オンライン学習
group_study = np.random.normal(loc=75, scale=9, size=30)    # グループ学習

# DataFrameにまとめる
df = pd.DataFrame({
    "score": np.concatenate([lecture, online, group_study]),
    "method": ["通常講義"] * 30 + ["オンライン"] * 30 + ["グループ学習"] * 30,
    "gender": ["男性"] * 15 + ["女性"] * 15 + ["男性"] * 15 + ["女性"] * 15 + ["男性"] * 15 + ["女性"] * 15
})

# 全ペアのt検定 + Bonferroni補正
groups = {"通常講義": lecture, "オンライン": online, "グループ学習": group_study}
pairs = list(combinations(groups.keys(), 2))
n_comparisons = len(pairs)
alpha_corrected = 0.05 / n_comparisons  # Bonferroni補正

print(f"=== Bonferroni補正による多重比較 ===")
print(f"比較回数: {n_comparisons}")
print(f"補正後の有意水準: {alpha_corrected:.4f}\n")

print(f"{'ペア':<25} {'t値':>8} {'p値':>10} {'判定':<10}")
print("-" * 60)

for g1, g2 in pairs:
    t_stat, p_val = stats.ttest_ind(groups[g1], groups[g2])
    significant = "有意差あり" if p_val < alpha_corrected else "有意差なし"
    print(f"{g1} vs {g2:<12} {t_stat:>8.4f} {p_val:>10.6f} {significant}")