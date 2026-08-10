import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.stats import chi2_contingency


# ========================================
# サンプルデータ作成
# ========================================

df = pd.DataFrame({
    "season": [
        "春", "春", "春", "春", "春",
        "夏", "夏", "夏", "夏", "夏",
        "秋", "秋", "秋", "秋", "秋",
        "冬", "冬", "冬", "冬", "冬"
    ],
    "drink": [
        "コーヒー", "紅茶", "紅茶", "コーヒー", "ジュース",
        "ジュース", "ジュース", "お茶", "ジュース", "お茶",
        "コーヒー", "紅茶", "コーヒー", "お茶", "紅茶",
        "コーヒー", "コーヒー", "紅茶", "コーヒー", "紅茶"
    ]
})


# ========================================
# クロス集計表
# ========================================

cross_table = pd.crosstab(
    df["season"],
    df["drink"]
)

print("【クロス集計表】")
print(cross_table)


# ========================================
# クラメールのVを計算
# ========================================

chi2, p, dof, expected = chi2_contingency(cross_table)

# データ件数
n = cross_table.values.sum()

# 行数と列数
rows, cols = cross_table.shape

# クラメールのV
cramers_v = np.sqrt(
    chi2 / (n * min(rows - 1, cols - 1))
)


print("\n【カイ二乗検定】")
print(f"カイ二乗値: {chi2:.3f}")
print(f"p値: {p:.4f}")

print("\n【クラメールのV】")
print(f"V = {cramers_v:.3f}")


# ========================================
# グラフ作成
# ========================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)


# ========================================
# ヒートマップ
# ========================================

sns.heatmap(
    cross_table,
    annot=True,
    fmt="d",
    cmap="YlOrRd",
    ax=axes[0]
)

axes[0].set_title(
    "クロス集計表（ヒートマップ）",
    fontsize=13
)

axes[0].set_xlabel("ドリンク")
axes[0].set_ylabel("季節")


# ========================================
# 積み上げ棒グラフ
# ========================================

cross_table.plot(
    kind="bar",
    stacked=True,
    ax=axes[1],
    colormap="Set2"
)

axes[1].set_title(
    f"季節別ドリンク構成（V={cramers_v:.3f}）",
    fontsize=13
)

axes[1].set_xlabel("季節")
axes[1].set_ylabel("人数")

axes[1].legend(
    title="ドリンク"
)

axes[1].tick_params(
    axis="x",
    rotation=0
)


# ========================================
# グラフ表示
# ========================================

plt.tight_layout()
plt.show()