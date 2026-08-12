import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

x = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Step関数
axes[0,0].plot(x, np.where(x > 0, 1, 0), linewidth=2, color='steelblue')
axes[0,0].set_title("Step（ステップ関数）", fontsize=11)
axes[0,0].set_ylim(-0.1, 1.1)
axes[0,0].grid(alpha=0.3)
axes[0,0].axhline(0, color='gray', linewidth=0.5)
axes[0,0].axvline(0, color='gray', linewidth=0.5)

# Sigmoid関数
sigmoid = 1 / (1 + np.exp(-x))
axes[0,1].plot(x, sigmoid, linewidth=2, color='steelblue')
axes[0,1].set_title("Sigmoid（シグモイド関数）", fontsize=11)
axes[0,1].grid(alpha=0.3)
axes[0,1].axhline(0.5, color='red', linestyle='--', alpha=0.5)
axes[0,1].axvline(0, color='gray', linewidth=0.5)

# tanh関数
axes[0,2].plot(x, np.tanh(x), linewidth=2, color='steelblue')
axes[0,2].set_title("tanh（ハイパボリックタンジェント）", fontsize=11)
axes[0,2].grid(alpha=0.3)
axes[0,2].axhline(0, color='gray', linewidth=0.5)
axes[0,2].axvline(0, color='gray', linewidth=0.5)

# ReLU関数
axes[1,0].plot(x, np.maximum(0, x), linewidth=2, color='steelblue')
axes[1,0].set_title("ReLU（ランプ関数）", fontsize=11)
axes[1,0].grid(alpha=0.3)
axes[1,0].axhline(0, color='gray', linewidth=0.5)
axes[1,0].axvline(0, color='gray', linewidth=0.5)

# Softmax（概念的表示）
axes[1,1].bar(range(5), [0.05, 0.1, 0.6, 0.15, 0.1], color='steelblue', alpha=0.8)
axes[1,1].set_title("Softmax（出力例: 合計=1）", fontsize=11)
axes[1,1].set_xlabel("クラス")
axes[1,1].set_ylabel("確率")
axes[1,1].grid(axis='y', alpha=0.3)

# 比較
axes[1,2].plot(x, sigmoid, linewidth=2, label='Sigmoid')
axes[1,2].plot(x, np.tanh(x), linewidth=2, label='tanh')
axes[1,2].plot(x, np.maximum(0, x), linewidth=2, label='ReLU')
axes[1,2].set_title("活性化関数の比較", fontsize=11)
axes[1,2].legend()
axes[1,2].grid(alpha=0.3)
axes[1,2].set_ylim(-1.5, 5)

plt.tight_layout()
plt.show()