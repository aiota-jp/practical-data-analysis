import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib

# データの準備
np.random.seed(42)
lecture = np.random.normal(loc=65, scale=10, size=30)       # 通常講義
online = np.random.normal(loc=70, scale=12, size=30)        # オンライン学習
group_study = np.random.normal(loc=75, scale=9, size=30)    # グループ学習

# DataFrameにまとめる
df = pd.DataFrame({
    "score": np.concatenate([lecture, online, group_study]),
    "method": ["通常講義"]*30 + ["オンライン"]*30 + ["グループ学習"]*30
})

# 各グループの平均
mean_all = df["score"].mean()                    # 全体平均
mean_lecture = lecture.mean()                      # 通常講義の平均
mean_online = online.mean()                       # オンラインの平均
mean_group = group_study.mean()                   # グループ学習の平均

# 平方和の計算
# 総平方和（SST）
sst = np.sum((df["score"] - mean_all) ** 2)

# 群間平方和（SSB）
ssb = (30 * (mean_lecture - mean_all)**2 +
       30 * (mean_online - mean_all)**2 +
       30 * (mean_group - mean_all)**2)

# 群内平方和（SSE）
sse = (np.sum((lecture - mean_lecture)**2) +
       np.sum((online - mean_online)**2) +
       np.sum((group_study - mean_group)**2))

# 自由度
df_between = 3 - 1          # k - 1 = 2
df_within = 90 - 3          # N - k = 87

# 平均平方（分散）
msb = ssb / df_between      # 群間平均平方
mse = sse / df_within       # 群内平均平方

# F値
f_value = msb / mse

# p値
p_value = 1 - stats.f.cdf(f_value, df_between, df_within)

print(f"\n=== 分散分析表（手動計算） ===")
print(f"{'要因':<10} {'平方和':>10} {'自由度':>6} {'平均平方':>10} {'F値':>8} {'p値':>10}")
print("-" * 60)
print(f"{'群間':<10} {ssb:>10.2f} {df_between:>6} {msb:>10.2f} {f_value:>8.4f} {p_value:>10.6f}")
print(f"{'群内':<10} {sse:>10.2f} {df_within:>6} {mse:>10.2f}")
print(f"{'全体':<10} {sst:>10.2f} {89:>6}")