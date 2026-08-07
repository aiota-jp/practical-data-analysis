import pandas as pd
import numpy as np
from scipy import stats

# サンプルデータの作成
df = pd.DataFrame({
    "value": [
        10, 11, 12, 13, 14,
        10, 11, 12, 13, 14,
        10, 11, 12, 13, 14,
        10, 11, 12, 13, 14,
        100
    ]
})

print("===== 元のデータ =====")
print(df)


# ========================================
# Zスコアの計算
# ========================================

z_scores = stats.zscore(df["value"])

df["z_score"] = z_scores

print("\n===== Zスコア =====")
print(df)


# ========================================
# |Z| > 3 のデータを外れ値として検出
# ========================================

outliers = df[np.abs(df["z_score"]) > 3]

print("\n===== 外れ値 =====")
print(outliers)