import pandas as pd
import numpy as np

df = pd.DataFrame({
    "value": [10, 12, 14, 15, 20, 30, 50, 100, 500, 1000]
})

print("===== 変換前 =====")
print(df)

# 対数変換
df["value_log"] = np.log1p(df["value"])

print("\n===== 対数変換後 =====")
print(df)