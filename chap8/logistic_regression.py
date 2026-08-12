import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import japanize_matplotlib

# サンプルデータ（商品購入予測）
np.random.seed(42)
n = 200

age = np.random.uniform(20, 60, n)
income = np.random.uniform(200, 1000, n)

# 購入確率はageとincomeに依存
logit = -5 + 0.05 * age + 0.005 * income
prob = 1 / (1 + np.exp(-logit))
purchase = np.random.binomial(1, prob, n)

df_log = pd.DataFrame({"age": age, "income": income, "purchase": purchase})

# ロジスティック回帰（statsmodels）
model_logit = smf.logit("purchase ~ age + income", data=df_log).fit()

print("=== ロジスティック回帰の結果 ===")
print(model_logit.summary())

# オッズ比の表示
print(f"\n=== オッズ比 ===")
odds_ratios = np.exp(model_logit.params)
for var in ["age", "income"]:
    print(f"  {var}: {odds_ratios[var]:.4f}")
    print(f"    → {var}が1単位増えると、購入のオッズが{(odds_ratios[var]-1)*100:.2f}%変化")