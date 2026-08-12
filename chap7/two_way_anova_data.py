import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

# データの準備
np.random.seed(42)

# 各条件のデータ生成（学習方法 × 性別）
data = []
methods = ["通常講義", "オンライン", "グループ学習"]
genders = ["男性", "女性"]

# 効果の設定
method_effects = {"通常講義": 0, "オンライン": 5, "グループ学習": 10}
gender_effects = {"男性": 0, "女性": 3}
# 交互作用：グループ学習は女性により効果的
interaction = {("グループ学習", "女性"): 5}

for method in methods:
    for gender in genders:
        base = 60
        effect = base + method_effects[method] + gender_effects[gender]
        effect += interaction.get((method, gender), 0)
        scores = np.random.normal(loc=effect, scale=8, size=15)
        for score in scores:
            data.append({"method": method, "gender": gender, "score": score})

df = pd.DataFrame(data)

# 基本統計量
print("=== グループ別基本統計量 ===")
summary = df.groupby(["method", "gender"])["score"].agg(["count", "mean", "std"])
print(summary.round(2))