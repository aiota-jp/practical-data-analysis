import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM

# 例: 同じ10人の被験者が3つの条件でテストを受けた
np.random.seed(42)
n_subjects = 10

df_repeated = pd.DataFrame({
    "subject": list(range(n_subjects)) * 3,
    "condition": ["条件A"]*n_subjects + ["条件B"]*n_subjects + ["条件C"]*n_subjects,
    "score": np.concatenate([
        np.random.normal(60, 8, n_subjects),   # 条件A
        np.random.normal(65, 8, n_subjects),   # 条件B
        np.random.normal(70, 8, n_subjects),   # 条件C
    ])
})

# 反復測定分散分析
rm_anova = AnovaRM(df_repeated, depvar='score', subject='subject', within=['condition'])
result = rm_anova.fit()
print("=== 反復測定分散分析 ===")
print(result)