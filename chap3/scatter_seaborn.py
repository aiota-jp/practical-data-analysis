import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

# サンプルデータ
np.random.seed(42)
study_hours = np.random.uniform(1, 10, 50)
test_scores = 40 + study_hours * 5 + np.random.normal(0, 5, 50)
gender = np.random.choice(["男性", "女性"], 50)

# DataFrameの作成
df = pd.DataFrame({
    "勉強時間": study_hours,
    "テストスコア": test_scores,
    "性別": gender
})

print("===== サンプルデータ =====")
print(df.head(10))

# カテゴリ別の散布図
plt.figure(figsize=(8, 6))
sns.scatterplot(x="勉強時間", y="テストスコア", hue="性別", data=df, s=80)
plt.title("性別ごとの勉強時間とスコアの関係", fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# 回帰直線付き散布図
sns.lmplot(x="勉強時間", y="テストスコア", hue="性別", data=df, height=6, aspect=1.2)
plt.title("回帰直線付き散布図")
plt.show()