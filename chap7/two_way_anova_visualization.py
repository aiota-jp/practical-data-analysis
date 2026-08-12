import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# データの準備
np.random.seed(42)
lecture = np.random.normal(loc=65, scale=10, size=30)       # 通常講義
online = np.random.normal(loc=70, scale=12, size=30)        # オンライン学習
group_study = np.random.normal(loc=75, scale=9, size=30)    # グループ学習

# DataFrameにまとめる
df = pd.DataFrame({
    "score": np.concatenate([lecture, online, group_study]),
    "method": ["通常講義"] * 30 + ["オンライン"] * 30 + ["グループ学習"] * 30,
    "gender": ["男性"] * 15 + ["女性"] * 15 + ["男性"] * 15 + ["女性"] * 15 + ["男性"] * 15 + ["女性"] * 15
})

# 交互作用プロット
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# プロット1: 学習方法×性別の平均スコア
means = df.groupby(["method", "gender"])["score"].mean().unstack()
means = means.reindex(["通常講義", "オンライン", "グループ学習"])
means.plot(marker='o', linewidth=2, markersize=8, ax=axes[0])
axes[0].set_title("交互作用プロット（学習方法×性別）", fontsize=12)
axes[0].set_xlabel("学習方法")
axes[0].set_ylabel("平均スコア")
axes[0].legend(title="性別")
axes[0].grid(alpha=0.3)

# プロット2: 箱ひげ図
sns.boxplot(x="method", y="score", hue="gender", data=df, ax=axes[1],
            palette="Set2", order=["通常講義", "オンライン", "グループ学習"])
axes[1].set_title("学習方法×性別のスコア分布", fontsize=12)
axes[1].set_xlabel("学習方法")
axes[1].set_ylabel("スコア")
axes[1].legend(title="性別")

plt.tight_layout()
plt.show()