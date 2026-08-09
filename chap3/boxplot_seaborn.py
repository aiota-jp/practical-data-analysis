import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import pandas as pd

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

# DataFrameでの箱ひげ図
df = pd.DataFrame({
    "score": np.concatenate([data_a, data_b, data_c]),
    "group": ["A"]*100 + ["B"]*100 + ["C"]*100
})

plt.figure(figsize=(8, 6))
sns.boxplot(x="group", y="score", data=df, palette="Set2")
plt.xlabel("グループ", fontsize=12)
plt.ylabel("スコア", fontsize=12)
plt.title("グループ別スコアの箱ひげ図", fontsize=14)
plt.show()
