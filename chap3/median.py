import pandas as pd
import numpy as np

data = [300, 350, 400, 420, 450, 480, 500, 550, 600, 3000]

median_val = np.median(data)
print(f"中央値: {median_val}")  # 465.0（外れ値の影響を受けない）
