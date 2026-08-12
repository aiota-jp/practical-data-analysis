import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib

# 4つの広告パターンのクリック率データ
np.random.seed(42)
ad_A = np.random.normal(loc=3.2, scale=0.8, size=50)   # パターンA
ad_B = np.random.normal(loc=3.5, scale=0.9, size=50)   # パターンB
ad_C = np.random.normal(loc=4.1, scale=0.7, size=50)   # パターンC
ad_D = np.random.normal(loc=3.3, scale=1.0, size=50)   # パターンD

df_ad = pd.DataFrame({
    "ctr": np.concatenate([ad_A, ad_B, ad_C, ad_D]),
    "pattern": ["A"]*50 + ["B"]*50 + ["C"]*50 + ["D"]*50
})

# 効果量の計算
all_scores = np.concatenate([ad_A, ad_B, ad_C, ad_D])
grand_mean = all_scores.mean()

ssb = sum(50 * (g.mean() - grand_mean)**2 for g in [ad_A, ad_B, ad_C, ad_D])
sst = np.sum((all_scores - grand_mean)**2)

eta_squared = ssb / sst
print(f"\n=== 効果量 ===")
print(f"η² = {eta_squared:.4f}")

if eta_squared < 0.01:
    print("→ 効果量: ほとんどなし")
elif eta_squared < 0.06:
    print("→ 効果量: 小")
elif eta_squared < 0.14:
    print("→ 効果量: 中")
else:
    print("→ 効果量: 大")