# AICによる変数選択
from itertools import combinations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# サンプルデータ（住宅データ）
np.random.seed(42)
n = 100
area = np.random.uniform(40, 120, n)       # 面積（㎡）
distance = np.random.uniform(1, 30, n)     # 駅距離（分）
age = np.random.uniform(0, 40, n)          # 築年数
rooms = np.random.randint(1, 5, n)         # 部屋数

# 住宅価格（万円）
price = 30 * area - 50 * distance - 20 * age + 200 * rooms + 1000 + np.random.normal(0, 300, n)

# DataFrameの作成
df = pd.DataFrame({"price": price, "area": area, "distance": distance, "age": age, "rooms": rooms})

# AICによる変数選択
variables = ["area", "distance", "age", "rooms"]
best_aic = np.inf
best_model_vars = None

print("=== AICによるモデル選択 ===")
for k in range(1, len(variables) + 1):
    for combo in combinations(variables, k):
        formula = f"price ~ {' + '.join(combo)}"
        m = smf.ols(formula, data=df).fit()
        if m.aic < best_aic:
            best_aic = m.aic
            best_model_vars = combo

print(f"最良モデルの変数: {best_model_vars}")
print(f"最良モデルのAIC: {best_aic:.2f}")