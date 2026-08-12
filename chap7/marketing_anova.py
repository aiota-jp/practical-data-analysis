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

# 1. 分散分析
f_stat, p_value = stats.f_oneway(ad_A, ad_B, ad_C, ad_D)
print(f"=== 広告パターン別CTRの分散分析 ===")
print(f"F値: {f_stat:.4f}, p値: {p_value:.6f}")

# 2. 多重比較
if p_value < 0.05:
    tukey = pairwise_tukeyhsd(df_ad["ctr"], df_ad["pattern"], alpha=0.05)
    print(f"\n=== Tukey HSD 多重比較 ===")
    print(tukey)

# 3. 可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(x="pattern", y="ctr", data=df_ad, ax=axes[0], palette="Set3")
axes[0].set_title(f"広告パターン別CTR\nF={f_stat:.2f}, p={p_value:.4f}", fontsize=12)
axes[0].set_xlabel("広告パターン")
axes[0].set_ylabel("CTR (%)")

# 平均値の棒グラフ + 信頼区間
means = df_ad.groupby("pattern")["ctr"].mean()
sems = df_ad.groupby("pattern")["ctr"].sem()
axes[1].bar(means.index, means.values, yerr=sems.values*1.96,
            capsize=5, color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'], alpha=0.8)
axes[1].set_title("平均CTRと95%信頼区間", fontsize=12)
axes[1].set_xlabel("広告パターン")
axes[1].set_ylabel("平均CTR (%)")
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()