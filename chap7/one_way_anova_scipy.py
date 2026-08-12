import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib

# データの準備
np.random.seed(42)
lecture = np.random.normal(loc=65, scale=10, size=30)       # 通常講義
online = np.random.normal(loc=70, scale=12, size=30)        # オンライン学習
group_study = np.random.normal(loc=75, scale=9, size=30)    # グループ学習

# DataFrameにまとめる
df = pd.DataFrame({
    "score": np.concatenate([lecture, online, group_study]),
    "method": ["通常講義"]*30 + ["オンライン"]*30 + ["グループ学習"]*30
})

# 一元分散分析の実行
f_stat, p_value = stats.f_oneway(lecture, online, group_study)

print(f"\n=== scipy.stats.f_oneway による一元分散分析 ===")
print(f"F値: {f_stat:.4f}")
print(f"p値: {p_value:.6f}")

alpha = 0.05
if p_value < alpha:
    print(f"\n→ p値({p_value:.6f}) < α({alpha})")
    print("  帰無仮説を棄却。学習方法によって試験成績に有意な差がある。")
    print("  → 次のステップ: 多重比較でどのペアに差があるか特定する")
else:
    print(f"\n→ p値({p_value:.6f}) ≥ α({alpha})")
    print("  帰無仮説を棄却できない。学習方法による差は認められない。")