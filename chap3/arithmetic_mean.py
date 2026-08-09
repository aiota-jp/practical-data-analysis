import numpy as np

data = [300, 350, 400, 420, 450, 480, 500, 550, 600, 3000]

# 算術平均
mean_val = np.mean(data)
print(f"算術平均: {mean_val}")  # 705.0（外れ値3000の影響で高くなる）
