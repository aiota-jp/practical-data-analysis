import pandas as pd
from sklearn.preprocessing import RobustScaler

# 外れ値を含むサンプルデータ
df = pd.DataFrame({
    "age": [25, 30, 35, 40, 45, 120],
    "salary": [
        3000000,
        4500000,
        5000000,
        7000000,
        8000000,
        50000000
    ]
})

print("===== 変換前 =====")
print(df)


# ========================================
# RobustScalerによるスケーリング
# ========================================

scaler = RobustScaler()

df_robust = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)

print("\n===== RobustScalerによる変換後 =====")
print(df_robust)