import numpy as np
from scipy import stats
import pandas as pd

# クロス集計表（観測値）
observed = np.array([[45, 455],   # デザインA: CVあり, CVなし
                     [63, 437]])  # デザインB: CVあり, CVなし

# カイ二乗検定
chi2, p_value, dof, expected = stats.chi2_contingency(observed)

print(f"=== カイ二乗検定（A/Bテスト） ===")
print(f"\n観測値:")
print(f"  デザインA: CVR = {45/500*100:.1f}% ({45}件/{500}件)")
print(f"  デザインB: CVR = {63/500*100:.1f}% ({63}件/{500}件)")

print(f"\n期待値（差がないと仮定した場合）:")
print(f"  デザインA: CVあり={expected[0,0]:.1f}, CVなし={expected[0,1]:.1f}")
print(f"  デザインB: CVあり={expected[1,0]:.1f}, CVなし={expected[1,1]:.1f}")

print(f"\n検定結果:")
print(f"  カイ二乗値: {chi2:.4f}")
print(f"  自由度: {dof}")
print(f"  p値: {p_value:.6f}")

alpha = 0.05
if p_value < alpha:
    print(f"\n→ p値({p_value:.6f}) < α({alpha}): 帰無仮説を棄却")
    print("  デザインAとBのコンバージョン率には統計的に有意な差がある。")
    print(f"  デザインBの方がCVRが高い（{63/500*100:.1f}% vs {45/500*100:.1f}%）")
else:
    print(f"\n→ p値({p_value:.6f}) ≥ α({alpha}): 帰無仮説を棄却できない")
    print("  デザインAとBのCVRに有意な差があるとは言えない。")