import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# サンプルデータ（1日の来店客数）
np.random.seed(42)
n = 100
ad_spend = np.random.uniform(10, 100, n)       # 広告費（万円）
is_holiday = np.random.binomial(1, 0.3, n)     # 休日フラグ

# 来店客数（ポアソン分布に従う）
log_lambda = 3.5 + 0.01 * ad_spend + 0.3 * is_holiday
visitors = np.random.poisson(np.exp(log_lambda))

# DataFrameの作成
df_poisson = pd.DataFrame({"visitors": visitors, "ad_spend": ad_spend, "is_holiday": is_holiday})

# ポアソン回帰
model_poisson = smf.glm("visitors ~ ad_spend + is_holiday", data=df_poisson, family=sm.families.Poisson()).fit()

print("=== ポアソン回帰の結果 ===")
print(model_poisson.summary())

# 予測例
print("\n=== 予測例 ===")
new_data = pd.DataFrame({"ad_spend": [50, 80], "is_holiday": [0, 1]})
predicted = model_poisson.predict(new_data)

for i, row in new_data.iterrows():
    print(f"広告費{row['ad_spend']:.0f}万円, 休日={'Yes' if row['is_holiday'] else 'No'} → 予測来店数: {predicted[i]:.0f}人")