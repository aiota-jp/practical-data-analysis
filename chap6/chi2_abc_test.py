import numpy as np
from scipy import stats
import pandas as pd

# 3パターン（A/B/C）のテスト結果
data = {
    "デザイン": ["A", "B", "C"],
    "CVあり": [45, 63, 58],
    "CVなし": [455, 437, 442],
    "合計": [500, 500, 500]
}
df = pd.DataFrame(data)
print("=== A/B/Cテスト結果 ===")
print(df.to_string(index=False))

# クロス集計表
observed = np.array([[45, 455],
                     [63, 437],
                     [58, 442]])

chi2, p_value, dof, expected = stats.chi2_contingency(observed)

print(f"\nカイ二乗値: {chi2:.4f}")
print(f"自由度: {dof}")
print(f"p値: {p_value:.6f}")

if p_value < 0.05:
    print("→ 3つのデザイン間でCVRに有意な差がある")
    # どのペアに差があるかは多重比較が必要（第7回で解説）
else:
    print("→ 3つのデザイン間でCVRに有意な差があるとは言えない")