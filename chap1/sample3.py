import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

# 折れ線グラフ
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.figure(figsize=(8,5))
plt.plot(x, y, marker="o", label="売上推移")
plt.xlabel("月")
plt.ylabel("売上（万円）")
plt.title("月別売上推移")
plt.legend()
plt.grid(True)
plt.show()

# ヒストグラム
data = np.random.randn(1000)

plt.figure(figsize=(8,5))
plt.hist(data, bins=30, edgecolor="black")
plt.xlabel("値")
plt.ylabel("頻度")
plt.title("データの分布")
plt.show()