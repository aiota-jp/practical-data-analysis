import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# データの準備
np.random.seed(42)
lecture = np.random.normal(loc=65, scale=10, size=30)       # 通常講義
online = np.random.normal(loc=70, scale=12, size=30)        # オンライン学習
group_study = np.random.normal(loc=75, scale=9, size=30)    # グループ学習

# DataFrameにまとめる
df = pd.DataFrame({
    "score": np.concatenate([lecture, online, group_study]),
    "method": ["通常講義"] * 30 + ["オンライン"] * 30 + ["グループ学習"] * 30,
    "gender": ["男性"] * 15 + ["女性"] * 15 + ["男性"] * 15 + ["女性"] * 15 + ["男性"] * 15 + ["女性"] * 15
})

# 二元配置分散分析
# ols: 線形モデル（最小二乗法）
# C(): カテゴリ変数であることを明示
# C(method):C(gender): 学習方法と性別の交互作用
model = ols("score ~ C(method) + C(gender) + C(method):C(gender)", data=df).fit()

# 分散分析表の作成（Type II）
anova_table = sm.stats.anova_lm(model, typ=2)

print("=== 二元配置分散分析表 ===")
print(anova_table.round(4))

# 結果の解釈
alpha = 0.05
print("\n=== 結果の解釈（有意水準 α=0.05） ===")

effects = {
    "C(method)": "学習方法の主効果",
    "C(gender)": "性別の主効果",
    "C(method):C(gender)": "交互作用効果"
}

for source, name in effects.items():
    p = anova_table.loc[source, "PR(>F)"]
    f_val = anova_table.loc[source, "F"]
    if p < alpha:
        print(f"{name}: F={f_val:.4f}, p={p:.6f} → 有意（p < 0.05）")
    else:
        print(f"{name}: F={f_val:.4f}, p={p:.6f} → 有意でない")