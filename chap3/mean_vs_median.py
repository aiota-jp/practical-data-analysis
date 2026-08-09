import pandas as pd
import numpy as np

# 年収データ（右に裾が長い分布）
salaries = [300, 320, 350, 380, 400, 420, 450, 500, 550, 600,
            650, 700, 800, 1000, 1500, 2000, 5000]

print(f"平均年収: {np.mean(salaries):.0f}万円")    # 約878万円
print(f"中央年収: {np.median(salaries):.0f}万円")  # 550万円

# → 年収のような偏った分布では中央値の方が「実感」に近い
