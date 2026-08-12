import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import japanize_matplotlib

# 先ほどの一元分散分析のデータを使用
np.random.seed(42)
lecture = np.random.normal(loc=65, scale=10, size=30)
online = np.random.normal(loc=70, scale=12, size=30)
group_study = np.random.normal(loc=75, scale=9, size=30)

scores = np.concatenate([lecture, online, group_study])
methods = ["通常講義"]*30 + ["オンライン"]*30 + ["グループ学習"]*30

# まず分散分析で有意差を確認
f_stat, p_value = stats.f_oneway(lecture, online, group_study)
print(f"=== 一元分散分析 ===")
print(f"F値: {f_stat:.4f}, p値: {p_value:.6f}")

if p_value < 0.05:
    print("→ 有意差あり。多重比較を実施します。\n")

# Tukey HSD法による多重比較
tukey_result = pairwise_tukeyhsd(scores, methods, alpha=0.05)
print("=== Tukey HSD 多重比較結果 ===")
print(tukey_result)

# 結果の可視化
fig = tukey_result.plot_simultaneous()
plt.title("Tukey HSD: グループ間の平均差と信頼区間", fontsize=12)
plt.xlabel("スコア")
plt.tight_layout()
plt.show()