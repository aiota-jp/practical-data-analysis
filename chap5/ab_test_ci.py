import numpy as np
from scipy import stats

# Webサイトのコンバージョン率のA/Bテスト
# パターンA: 1000人中120人がコンバージョン
# パターンB: 1000人中150人がコンバージョン

n_a, k_a = 1000, 120
n_b, k_b = 1000, 150

p_a = k_a / n_a
p_b = k_b / n_b

# 各パターンの95%信頼区間
z = stats.norm.ppf(0.975)

se_a = np.sqrt(p_a * (1 - p_a) / n_a)
ci_a = (p_a - z * se_a, p_a + z * se_a)

se_b = np.sqrt(p_b * (1 - p_b) / n_b)
ci_b = (p_b - z * se_b, p_b + z * se_b)

# 差の信頼区間
diff = p_b - p_a
se_diff = np.sqrt(se_a**2 + se_b**2)
ci_diff = (diff - z * se_diff, diff + z * se_diff)

print(f"=== A/Bテストの結果 ===")
print(f"パターンA: CVR={p_a*100:.1f}%, 95%CI=[{ci_a[0]*100:.1f}%, {ci_a[1]*100:.1f}%]")
print(f"パターンB: CVR={p_b*100:.1f}%, 95%CI=[{ci_b[0]*100:.1f}%, {ci_b[1]*100:.1f}%]")
print(f"差 (B-A): {diff*100:.1f}%, 95%CI=[{ci_diff[0]*100:.1f}%, {ci_diff[1]*100:.1f}%]")
print()

if ci_diff[0] > 0:
    print("→ 差の信頼区間が0を含まないため、BはAより有意に高いと判断できます。")
else:
    print("→ 差の信頼区間が0を含むため、有意な差があるとは断言できません。")