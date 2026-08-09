import numpy as np
from scipy import stats

# サンプルデータの作成（正規分布）
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=10, size=50)

print("===== サンプルデータ =====")
print(normal_data)

# シャピロ・ウィルク検定
stat, p_value = stats.shapiro(normal_data)

print("\n===== シャピロ・ウィルク検定 =====")
print(f"統計量: {stat:.4f}")
print(f"p値: {p_value:.4f}")

# 検定結果
print("\n===== 判定結果 =====")

if p_value > 0.05:
    print("→ 正規分布に従わないとはいえない（帰無仮説を棄却できない）")
else:
    print("→ 正規分布に従わないと判断する（帰無仮説を棄却）")