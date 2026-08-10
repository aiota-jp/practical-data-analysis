import numpy as np

# 標準誤差の計算
sample = np.array([165, 170, 172, 168, 175, 180, 162, 171, 169, 173])
n = len(sample)
se = sample.std(ddof=1) / np.sqrt(n)

print(f"標本平均: {sample.mean():.2f}")
print(f"不偏標準偏差: {sample.std(ddof=1):.2f}")
print(f"標準誤差 SE: {se:.2f}")
print(f"サンプルサイズ n: {n}")