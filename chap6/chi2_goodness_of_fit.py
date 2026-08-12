from scipy import stats
import numpy as np

# 例: サイコロを600回振った結果（公正なら各面100回ずつ出るはず）
observed_freq = np.array([90, 105, 95, 110, 85, 115])  # 実際の結果
expected_freq = np.array([100, 100, 100, 100, 100, 100])  # 期待値

# カイ二乗適合度検定
chi2_stat, p_value = stats.chisquare(observed_freq, expected_freq)

print(f"=== カイ二乗適合度検定 ===")
print(f"観測値: {observed_freq}")
print(f"期待値: {expected_freq}")
print(f"カイ二乗値: {chi2_stat:.4f}")
print(f"p値: {p_value:.6f}")

if p_value < 0.05:
    print("→ このサイコロは公正ではない可能性がある")
else:
    print("→ このサイコロは公正であると言える")