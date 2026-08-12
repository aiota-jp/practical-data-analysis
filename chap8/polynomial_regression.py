import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import japanize_matplotlib

# 非線形データの生成
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 5 + 3*x - 0.5*x**2 + np.random.normal(0, 3, 50)

df_poly = pd.DataFrame({"x": x, "y": y})

# 線形モデル（1次）
model_1 = smf.ols("y ~ x", data=df_poly).fit()

# 2次多項式モデル
model_2 = smf.ols("y ~ x + I(x**2)", data=df_poly).fit()

# 3次多項式モデル
model_3 = smf.ols("y ~ x + I(x**2) + I(x**3)", data=df_poly).fit()

# AICの比較
print("=== 多項式回帰のモデル比較 ===")
print(f"1次（線形）: R²={model_1.rsquared:.4f}, AIC={model_1.aic:.2f}")
print(f"2次（二次式）: R²={model_2.rsquared:.4f}, AIC={model_2.aic:.2f}")
print(f"3次（三次式）: R²={model_3.rsquared:.4f}, AIC={model_3.aic:.2f}")

# 可視化
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.7, color='steelblue', edgecolors='black', label='実測値')

x_plot = np.linspace(0, 10, 100)
df_plot = pd.DataFrame({"x": x_plot})

plt.plot(x_plot, model_1.predict(df_plot), 'g--', linewidth=2,
         label=f'1次 (R²={model_1.rsquared:.3f})')
plt.plot(x_plot, model_2.predict(df_plot), 'r-', linewidth=2,
         label=f'2次 (R²={model_2.rsquared:.3f})')
plt.plot(x_plot, model_3.predict(df_plot), 'purple', linewidth=2, linestyle=':',
         label=f'3次 (R²={model_3.rsquared:.3f})')

plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.title("多項式回帰の比較", fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.show()