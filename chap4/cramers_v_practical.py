import pandas as pd
import numpy as np
from scipy import stats

# より実践的なデータ
np.random.seed(42)
n = 200

# 季節と売れ筋ドリンクのデータを生成
seasons = np.random.choice(["春", "夏", "秋", "冬"], n)
drinks = []
for s in seasons:
    if s == "夏":
        drinks.append(np.random.choice(["アイスコーヒー", "ホットコーヒー", "お茶"], p=[0.6, 0.1, 0.3]))
    elif s == "冬":
        drinks.append(np.random.choice(["アイスコーヒー", "ホットコーヒー", "お茶"], p=[0.1, 0.6, 0.3]))
    else:
        drinks.append(np.random.choice(["アイスコーヒー", "ホットコーヒー", "お茶"], p=[0.3, 0.3, 0.4]))

df = pd.DataFrame({"季節": seasons, "ドリンク": drinks})

# クロス集計表
cross_table = pd.crosstab(df["季節"], df["ドリンク"])
print("クロス集計表:")
print(cross_table)

# カイ二乗検定とクラメールのV
chi2, p, dof, expected = stats.chi2_contingency(cross_table)
n_total = cross_table.sum().sum()
k = min(cross_table.shape) - 1
cramers_v = np.sqrt(chi2 / (n_total * k))

print(f"\nカイ二乗値: {chi2:.4f}")
print(f"p値: {p:.6f}")
print(f"クラメールのV: {cramers_v:.4f}")