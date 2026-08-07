import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import japanize_matplotlib

df = pd.read_csv("data.csv")

# 相関行列のヒートマップ
plt.figure(figsize=(10, 8))

# 数値データのみで相関係数を計算
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("相関行列")
plt.show()

# 箱ひげ図
sns.boxplot(x="category", y="value", data=df)
plt.title("カテゴリ別の値の分布")
plt.show()

# ペアプロット（全変数の組み合わせ散布図）
sns.pairplot(df, hue="category")
plt.show()