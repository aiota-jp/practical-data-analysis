import pandas as pd
import numpy as np

# 科目ごとの点数と単位数（重み）
scores = [80, 70, 90]       # 数学、英語、物理
weights = [4, 2, 3]          # 単位数

# 加重平均
weighted_mean = np.average(scores, weights=weights)
print(f"加重平均: {weighted_mean:.1f}")  # 81.1
