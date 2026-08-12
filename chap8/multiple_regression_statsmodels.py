# statsmodelsによる重回帰分析
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

print("=== データの概要 ===")
print(df.describe().round(2))

# 重回帰モデルの構築
model = smf.ols("price ~ area + distance + age + rooms", data=df).fit()

print("\n=== 重回帰分析の結果 ===")
print(model.summary())