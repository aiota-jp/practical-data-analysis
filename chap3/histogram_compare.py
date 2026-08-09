import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語表示対応

# 複数のデータを重ねて比較
np.random.seed(42)
class_a = np.random.normal(loc=70, scale=10, size=100)
class_b = np.random.normal(loc=60, scale=15, size=100)

plt.figure(figsize=(10, 6))
plt.hist(class_a, bins=15, alpha=0.6, label='クラスA', color='steelblue', edgecolor='black')
plt.hist(class_b, bins=15, alpha=0.6, label='クラスB', color='coral', edgecolor='black')
plt.xlabel("テストスコア", fontsize=12)
plt.ylabel("度数", fontsize=12)
plt.title("クラス別スコア分布の比較", fontsize=14)
plt.legend(fontsize=11)
plt.show()
