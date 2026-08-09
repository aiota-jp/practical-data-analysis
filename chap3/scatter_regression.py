import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# サンプルデータ
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
test_scores = np.array([42, 48, 55, 58, 65, 70, 74, 82, 88, 92])

# 散布図
plt.figure(figsize=(8, 6))
plt.scatter(study_hours, test_scores, alpha=0.7, color="steelblue", edgecolors="black")

# 回帰直線
z = np.polyfit(study_hours, test_scores, 1)
p = np.poly1d(z)
x_line = np.linspace(study_hours.min(), study_hours.max(), 100)
plt.plot(x_line, p(x_line), color="red", linestyle="--", linewidth=2, label=f"回帰直線: y = {z[0]:.2f}x + {z[1]:.2f}")

# グラフの設定
plt.xlabel("勉強時間（時間）", fontsize=12)
plt.ylabel("テストスコア", fontsize=12)
plt.title("勉強時間とテストスコアの関係（回帰直線付き）", fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.show()