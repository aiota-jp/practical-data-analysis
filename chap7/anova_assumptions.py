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

# 正規性の検定（シャピロ・ウィルク検定）
print("=== 正規性の検定（シャピロ・ウィルク検定） ===")
for name, data in [("通常講義", lecture), ("オンライン", online), ("グループ学習", group_study)]:
    stat, p = stats.shapiro(data)
    result = "正規分布" if p > 0.05 else "非正規分布"
    print(f"  {name}: 統計量={stat:.4f}, p値={p:.4f} → {result}")

# 等分散性の検定（Levene検定）
print("\n=== 等分散性の検定（Levene検定） ===")
stat, p = stats.levene(lecture, online, group_study)
print(f"  統計量={stat:.4f}, p値={p:.4f}")
if p > 0.05:
    print("  → 等分散を仮定できる（通常のANOVAが適用可能）")
else:
    print("  → 等分散を仮定できない（Welchの分散分析を使用）")