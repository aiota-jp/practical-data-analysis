import numpy as np
import pandas as pd
from scipy import stats

# クロス集計表の作成
observed = np.array([[43, 37],   # 理系: 好き, 嫌い
                     [10, 60]])  # 文系: 好き, 嫌い

# カイ二乗検定
chi2, p_value, dof, expected = stats.chi2_contingency(observed)

print(f"カイ二乗値: {chi2:.4f}")
print(f"p値: {p_value:.6f}")
print(f"自由度: {dof}")
print(f"期待値:\n{expected}")

# クラメールのVの計算
n = observed.sum()
k = min(observed.shape) - 1
cramers_v = np.sqrt(chi2 / (n * k))
print(f"\nクラメールのV: {cramers_v:.4f}")

# 判定
if cramers_v < 0.1:
    print("→ ほとんど相関がない")
elif cramers_v < 0.3:
    print("→ 弱い相関がある")
else:
    print("→ 強い相関がある")