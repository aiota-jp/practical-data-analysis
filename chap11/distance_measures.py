import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"ユークリッド距離: {euclidean(a, b):.4f}")
print(f"マンハッタン距離: {cityblock(a, b):.4f}")
print(f"コサイン距離: {cosine(a, b):.4f}")
print(f"コサイン類似度: {1 - cosine(a, b):.4f}")
