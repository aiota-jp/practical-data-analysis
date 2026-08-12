import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns

# データの準備
np.random.seed(42)
lecture = np.random.normal(loc=65, scale=10, size=30)       # 通常講義
online = np.random.normal(loc=70, scale=12, size=30)        # オンライン学習
group_study = np.random.normal(loc=75, scale=9, size=30)    # グループ学習

# DataFrameにまとめる
df = pd.DataFrame({
    "score": np.concatenate([lecture, online, group_study]),
    "method": ["通常講義"] * 30 + ["オンライン"] * 30 + ["グループ学習"] * 30
})

# 一元配置分散分析（ANOVA）
f_stat, p_value = stats.f_oneway(lecture, online, group_study)

print("=== 一元配置分散分析 ===")
print(f"F値 : {f_stat:.2f}")
print(f"p値 : {p_value:.4f}")

if p_value < 0.05:
    print("結果：学習方法によって平均点に有意な差があります。")
else:
    print("結果：学習方法による平均点の有意な差はありません。")

# グラフ作成
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
order = ["通常講義", "オンライン", "グループ学習"]

# 箱ひげ図
sns.boxplot(x="method", y="score", data=df, ax=axes[0], hue="method", palette="Set2", order=order, legend=False)
axes[0].set_title(f"学習方法別スコア分布\nF={f_stat:.2f}, p={p_value:.4f}", fontsize=12)
axes[0].set_xlabel("学習方法")
axes[0].set_ylabel("試験成績")
axes[0].axhline(df["score"].mean(), color="red", linestyle="--", alpha=0.5, label="全体平均")
axes[0].legend()

# ストリッププロット
sns.stripplot(x="method", y="score", data=df, ax=axes[1], hue="method", palette="Set2", order=order, alpha=0.6, jitter=True, legend=False)

# 各グループの平均値を赤線で表示
means = df.groupby("method")["score"].mean()
for i, method in enumerate(order):
    axes[1].hlines(means[method], i - 0.3, i + 0.3, color="red", linewidth=3)

axes[1].set_title("個別データ点と各グループの平均（赤線）", fontsize=12)
axes[1].set_xlabel("学習方法")
axes[1].set_ylabel("試験成績")

plt.tight_layout()
plt.show()