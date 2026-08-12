import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# サンプルデータ（気温とビール売上）
np.random.seed(42)
temperature = np.array([18, 20, 22, 24, 25, 26, 27, 28, 30, 32, 33, 35])
beer_sales = 50 + 8 * temperature + np.random.normal(0, 15, len(temperature))


# DataFrameの作成
df = pd.DataFrame({"temperature": temperature, "beer_sales": beer_sales})

# OLS（最小二乗法）による線形回帰
model = smf.ols("beer_sales ~ temperature", data=df).fit()

# 結果の表示
print(model.summary())