import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# サンプルデータ（住宅データ）
np.random.seed(42)
n = 100
area = np.random.uniform(40, 120, n)       # 面積（㎡）
distance = np.random.uniform(1, 30, n)     # 駅距離（分）
age = np.random.uniform(0, 40, n)          # 築年数
rooms = np.random.randint(1, 5, n)         # 部屋数

# DataFrameの作成
df = pd.DataFrame({"area": area, "distance": distance, "age": age, "rooms": rooms})

# VIFの計算
X = df[["area", "distance", "age", "rooms"]]
X = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["変数"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# 定数項を除外して表示
vif_data = vif_data[vif_data["変数"] != "const"]

print("=== VIF（分散膨張因子） ===")
print(vif_data.round(3))
print("\n※ VIFが10以上の場合、多重共線性が強い可能性があります。")