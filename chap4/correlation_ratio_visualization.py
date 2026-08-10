import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib

# ==============================
# サンプルデータ
# ==============================
df = pd.DataFrame({
    "age": [35, 19, 23, 18, 42, 54, 26, 21, 48, 61],
    "destination": [
        "北海道", "沖縄", "沖縄", "沖縄", "北海道",
        "北海道", "沖縄", "沖縄", "北海道", "北海道"
    ]
})


# ==============================
# 相関比 η² を計算する関数
# ==============================
def correlation_ratio(categories, values):
    # 全体の平均
    overall_mean = np.mean(values)

    # 全体の平方和
    total_ss = np.sum((values - overall_mean) ** 2)

    # グループ間平方和
    between_ss = 0

    for category in categories.unique():
        group = values[categories == category]

        between_ss += len(group) * (
            np.mean(group) - overall_mean
        ) ** 2

    # 相関比 η²
    eta2 = between_ss / total_ss

    return eta2


# ==============================
# 相関比 η² の計算
# ==============================
eta2 = correlation_ratio(
    df["destination"],
    df["age"]
)

print("相関比 η² =", eta2)


# ==============================
# グラフ作成
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))


# ------------------------------
# 箱ひげ図
# ------------------------------
sns.boxplot(
    x="destination",
    y="age",
    hue="destination",
    data=df,
    ax=axes[0],
    palette="Set2",
    legend=False
)

axes[0].set_title(
    f"旅行先別の年齢分布（箱ひげ図）\nη²={eta2:.3f}",
    fontsize=12
)

axes[0].set_xlabel("旅行先")
axes[0].set_ylabel("年齢")


# ------------------------------
# ストリップロット
# ------------------------------
sns.stripplot(
    x="destination",
    y="age",
    hue="destination",
    data=df,
    ax=axes[1],
    size=10,
    palette="Set2",
    jitter=True,
    legend=False
)

axes[1].set_title(
    f"旅行先別の年齢分布（ストリップロット）\nη²={eta2:.3f}",
    fontsize=12
)

axes[1].set_xlabel("旅行先")
axes[1].set_ylabel("年齢")


# ==============================
# グラフ表示
# ==============================
plt.tight_layout()
plt.show()