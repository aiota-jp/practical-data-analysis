import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib


# ========================================
# 相関比 η² を計算する関数
# ========================================
def correlation_ratio(categories, values):

    # 全体の平均
    overall_mean = np.mean(values)

    # 全体の平方和
    total_ss = np.sum((values - overall_mean) ** 2)

    # グループ間平方和
    between_ss = 0

    for category in categories.unique():

        # 部署ごとのデータを取得
        group = values[categories == category]

        # グループ間平方和を計算
        between_ss += len(group) * (
            np.mean(group) - overall_mean
        ) ** 2

    # 相関比 η²
    eta2 = between_ss / total_ss

    return eta2


# ========================================
# より実践的な例：部署と年収の関係
# ========================================

# 乱数を固定
np.random.seed(42)

# サンプルデータ作成
df_company = pd.DataFrame({
    "salary": np.concatenate([
        np.random.normal(500, 50, 30),   # 営業部
        np.random.normal(650, 80, 30),   # 開発部
        np.random.normal(550, 60, 30),   # マーケ部
    ]),
    "department":
        ["営業部"] * 30
        + ["開発部"] * 30
        + ["マーケ部"] * 30
})


# ========================================
# 相関比 η² の計算
# ========================================

eta2 = correlation_ratio(
    df_company["department"],
    df_company["salary"]
)

print(f"部署と年収の相関比: {eta2:.4f}")


# ========================================
# 部署ごとの平均年収
# ========================================

print("\n【部署ごとの平均年収】")

print(
    df_company.groupby("department")["salary"].mean()
)


# ========================================
# 可視化
# ========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="department",
    y="salary",
    hue="department",
    data=df_company,
    palette="Set3",
    legend=False
)

plt.title(
    f"部署別の年収分布（η²={eta2:.3f}）",
    fontsize=13
)

plt.xlabel("部署")
plt.ylabel("年収（万円）")

plt.tight_layout()
plt.show()