import pandas as pd
import numpy as np

# サンプルデータの作成
np.random.seed(42)
df = pd.DataFrame({
    "age": [25, 30, 35, 28, 42, 38, 45, 33, 29, 50],
    "salary": [300, 450, 500, 380, 700, 600, 800, 480, 350, 900],
    "score": [72, 85, 90, 68, 95, 88, 92, 78, 80, 98]
})

# describe()で基本統計量を一括表示
print(df.describe())
