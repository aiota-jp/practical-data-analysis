import numpy as np

# 配列の作成
arr = np.array([1, 2, 3, 4, 5])

# 基本的な統計量
print(f"平均: {np.mean(arr)}")        # 平均: 3.0
print(f"標準偏差: {np.std(arr)}")      # 標準偏差: 1.4142...
print(f"最大値: {np.max(arr)}")        # 最大値: 5
print(f"合計: {np.sum(arr)}")          # 合計: 15

# 2次元配列（行列）の作成
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print(f"行列の形状: {matrix.shape}")   # (3, 3)