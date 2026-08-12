from scipy import stats
import numpy as np

# マン・ホイットニーU検定（正規性を仮定しない）
group_a = np.array([3, 5, 7, 2, 8, 4, 6, 9, 1, 5])
group_b = np.array([6, 8, 10, 7, 9, 11, 5, 12, 8, 7])

stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
print(f"マン・ホイットニーU検定:")
print(f"  統計量: {stat:.4f}")
print(f"  p値: {p_value:.6f}")