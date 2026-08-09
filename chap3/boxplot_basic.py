import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# サンプルデータ
np.random.seed(42)

data_a = np.random.normal(
    loc=50,
    scale=10,
    size=100
)

data_b = np.random.normal(
    loc=60,
    scale=15,
    size=100
)

data_c = np.random.normal(
    loc=55,
    scale=8,
    size=100
)


# ========================================
# 基本的な箱ひげ図
# ========================================
plt.figure(figsize=(8, 6))

plt.boxplot(
    [data_a, data_b, data_c],
    tick_labels=["グループA", "グループB", "グループC"],
    patch_artist=True,
    boxprops=dict(facecolor="lightblue")
)

plt.ylabel("値", fontsize=12)
plt.title("グループ別の箱ひげ図", fontsize=14)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.show()