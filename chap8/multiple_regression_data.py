import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import japanize_matplotlib

# サンプルデータ（住宅データ）
np.random.seed(42)
n = 100

area = np.random.uniform(40, 120, n)          # 面積（㎡）
distance = np.random.uniform(1, 30, n)         # 駅距離（分）
age = np.random.uniform(0, 40, n)              # 築年数
rooms = np.random.randint(1, 5, n)             # 部屋数

# 住宅価格（万円）= 面積の影響 + 駅距離の影響 + 築年数の影響 + ノイズ
price = (30 * area - 50 * distance - 20 * age + 200 * rooms
         + 1000 + np.random.normal(0, 300, n))

df = pd.DataFrame({
    "price": price, "area": area, "distance": distance,
    "age": age, "rooms": rooms
})

print("=== データの概要 ===")
print(df.describe().round(2))