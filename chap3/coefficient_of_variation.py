import pandas as pd
import numpy as np

# 体重と身長のばらつきを比較
weight = np.array([55, 60, 65, 70, 75])
height = np.array([155, 160, 165, 170, 175])

cv_weight = (np.std(weight) / np.mean(weight)) * 100
cv_height = (np.std(height) / np.mean(height)) * 100

print(f"体重のCV: {cv_weight:.2f}%")
print(f"身長のCV: {cv_height:.2f}%")
# → CVが大きい方がばらつきが大きい
