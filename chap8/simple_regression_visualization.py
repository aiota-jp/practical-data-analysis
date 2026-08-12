import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import statsmodels.formula.api as smf

# サンプルデータ（気温とビール売上）
np.random.seed(42)
temperature = np.array([18, 20, 22, 24, 25, 26, 27, 28, 30, 32, 33, 35])
beer_sales = 50 + 8 * temperature + np.random.normal(0, 15, len(temperature))

# DataFrameの作成
df = pd.DataFrame({"temperature": temperature, "beer_sales": beer_sales})

# OLS（最小二乗法）による線形回帰
model = smf.ols("beer_sales ~ temperature", data=df).fit()

# 回帰係数を取得
b = model.params["Intercept"]       # 切片
a = model.params["temperature"]     # 傾き

# 結果表示
print("=== 単回帰分析 ===")
print(f"傾き: {a:.2f}")
print(f"切片: {b:.2f}")
print(f"回帰式: y = {a:.2f}x + {b:.2f}")
print(f"決定係数 R²: {model.rsquared:.4f}")

# 予測値の計算
x_pred = np.linspace(temperature.min() - 2, temperature.max() + 2, 100)
y_pred = model.predict(pd.DataFrame({"temperature": x_pred}))

# 可視化
plt.figure(figsize=(10, 6))
plt.scatter(temperature, beer_sales, s=80, color="steelblue", edgecolors="black", zorder=5, label="実測値")
plt.plot(x_pred, y_pred, color="red", linewidth=2, label=f"回帰直線: y = {a:.2f}x + {b:.2f}")

# 残差を可視化
y_hat = model.predict(df)
for i in range(len(temperature)):
    plt.plot([temperature[i], temperature[i]], [beer_sales[i], y_hat.iloc[i]], color="gray", linestyle="--", alpha=0.5)

plt.xlabel("気温（℃）", fontsize=12)
plt.ylabel("ビール売上（本）", fontsize=12)
plt.title(f"単回帰分析: 気温とビール売上\nR² = {model.rsquared:.4f}", fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.show()