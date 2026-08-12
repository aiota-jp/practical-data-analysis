import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy import stats
import statsmodels.formula.api as smf

# サンプルデータ
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 10 + 5 * x + 2 * x**2 + np.random.normal(0, 10, 100)
df = pd.DataFrame({"x": x, "y": y})

# 2次多項式回帰モデル
model_2 = smf.ols("y ~ x + I(x**2)", data=df).fit()

# 残差と予測値
residuals = model_2.resid
fitted = model_2.fittedvalues

# 残差診断
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 残差 vs 予測値（等分散性の確認）
axes[0,0].scatter(fitted, residuals, alpha=0.7, edgecolors="black")
axes[0,0].axhline(0, color="red", linestyle="--")
axes[0,0].set_xlabel("予測値")
axes[0,0].set_ylabel("残差")
axes[0,0].set_title("残差 vs 予測値（等分散性の確認）")

# 2. 残差のヒストグラム（正規性の確認）
axes[0,1].hist(residuals, bins=15, edgecolor="black", alpha=0.7, density=True)
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
axes[0,1].plot(x_norm, stats.norm.pdf(x_norm, residuals.mean(), residuals.std()), "r-", linewidth=2)
axes[0,1].set_xlabel("残差")
axes[0,1].set_ylabel("密度")
axes[0,1].set_title("残差のヒストグラム（正規性の確認）")

# 3. Q-Qプロット（正規性の確認）
stats.probplot(residuals, plot=axes[1,0])
axes[1,0].set_title("Q-Qプロット（正規性の確認）")

# 4. 残差の時系列プロット（独立性の確認）
axes[1,1].plot(residuals, "o-", alpha=0.7, markersize=4)
axes[1,1].axhline(0, color="red", linestyle="--")
axes[1,1].set_xlabel("観測番号")
axes[1,1].set_ylabel("残差")
axes[1,1].set_title("残差の時系列プロット（独立性の確認）")

plt.tight_layout()
plt.show()