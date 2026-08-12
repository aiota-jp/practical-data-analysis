import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns

# データの準備
np.random.seed(42)
lecture = np.random.normal(loc=65, scale=10, size=30)       # 通常講義
online = np.random.normal(loc=70, scale=12, size=30)        # オンライン学習
group_study = np.random.normal(loc=75, scale=9, size=30)    # グループ学習

# DataFrameにまとめる
df = pd.DataFrame({
    "score": np.concatenate([lecture, online, group_study]),
    "method": ["通常講義"] * 30 + ["オンライン"] * 30 + ["グループ学習"] * 30
})

# クラスカル・ウォリス検定（正規性を仮定しない）
stat, p = stats.kruskal(lecture, online, group_study)
print(f"\n=== クラスカル・ウォリス検定 ===")
print(f"統計量: {stat:.4f}, p値: {p:.6f}")