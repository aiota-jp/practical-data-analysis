import numpy as np

# 母集団（真の値）
np.random.seed(42)
population = np.random.normal(loc=170, scale=8, size=100000)  # 身長データ
print(f"=== 母集団の真の値 ===")
print(f"母平均 μ: {population.mean():.4f} cm")
print(f"母分散 σ²: {population.var():.4f}")
print(f"母標準偏差 σ: {population.std():.4f} cm")

# 標本を抽出して点推定
sample = np.random.choice(population, size=30, replace=False)
print(f"\n=== 標本からの点推定（n=30） ===")
print(f"標本平均 x̄（母平均の推定値）: {sample.mean():.4f} cm")
print(f"不偏分散 s²（母分散の推定値）: {sample.var(ddof=1):.4f}")
print(f"不偏標準偏差 s（母標準偏差の推定値）: {sample.std(ddof=1):.4f} cm")

# サンプルサイズを変えて精度を比較
print(f"\n=== サンプルサイズによる推定精度の変化 ===")
for n in [10, 30, 100, 500, 1000]:
    sample = np.random.choice(population, size=n, replace=False)
    print(f"n={n:5d}: 標本平均={sample.mean():.3f}, "
          f"誤差={abs(sample.mean() - population.mean()):.3f} cm")