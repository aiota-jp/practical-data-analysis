import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# サンプルデータ（勉強時間とテストスコア）
np.random.seed(42)
study_hours = np.random.uniform(1, 10, 50)
test_scores = 30 + 6 * study_hours + np.random.normal(0, 5, 50)

# 基本的な散布図
plt.figure(figsize=(8, 6))
plt.scatter(study_hours, test_scores, alpha=0.7, color='steelblue', edgecolors='black')
plt.xlabel("勉強時間（時間）", fontsize=12)
plt.ylabel("テストスコア", fontsize=12)
plt.title("勉強時間とテストスコアの関係", fontsize=14)
plt.grid(alpha=0.3)
plt.show()
