import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# AND（線形分離可能）
x_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0, 0, 0, 1])
axes[0].scatter(x_and[y_and==0,0], x_and[y_and==0,1], s=100, marker='o', label='0')
axes[0].scatter(x_and[y_and==1,0], x_and[y_and==1,1], s=100, marker='x', label='1')
axes[0].plot([0.5, 0.5, 1.5], [1.5, 0.5, 0.5], 'r--', linewidth=2)
axes[0].set_title("AND（線形分離可能）")
axes[0].legend()
axes[0].set_xlim(-0.5, 1.5)
axes[0].set_ylim(-0.5, 1.5)

# XOR（線形分離不可能）
x_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0, 1, 1, 0])
axes[1].scatter(x_xor[y_xor==0,0], x_xor[y_xor==0,1], s=100, marker='o', label='0')
axes[1].scatter(x_xor[y_xor==1,0], x_xor[y_xor==1,1], s=100, marker='x', label='1')
axes[1].set_title("XOR（線形分離不可能 → 多層パーセプトロンが必要）")
axes[1].legend()
axes[1].set_xlim(-0.5, 1.5)
axes[1].set_ylim(-0.5, 1.5)

plt.tight_layout()
plt.show()