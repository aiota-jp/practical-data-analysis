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

# 偏回帰係数
print("\n=== 偏回帰係数の解釈 ===")
for var in ["area", "distance", "age", "rooms"]:
    coef = model.params[var]
    p_val = model.pvalues[var]
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"{var:>10}: {coef:>8.2f} (p={p_val:.4f}) {sig}")

print(f"\nR²: {model.rsquared:.4f}")
print(f"調整済みR²: {model.rsquared_adj:.4f}")
print(f"AIC: {model.aic:.2f}")

# 結果の解釈
print("\n=== 解釈の例 ===")
print(f"面積が1㎡増えると、住宅価格は約{model.params['area']:.0f}万円上昇")
print(f"駅距離が1分遠くなると、住宅価格は約{abs(model.params['distance']):.0f}万円下落")