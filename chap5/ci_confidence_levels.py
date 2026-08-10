import numpy as np
from scipy import stats

sample = np.array([5.2, 4.8, 6.1, 5.5, 4.9, 5.7, 6.3, 5.0,
                   5.4, 4.6, 5.8, 6.0, 5.3, 4.7, 5.6])
n = len(sample)
x_bar = sample.mean()
se = sample.std(ddof=1) / np.sqrt(n)
df = n - 1

print("=== 信頼度別の信頼区間 ===")
for confidence in [0.90, 0.95, 0.99]:
    ci = stats.t.interval(confidence, df, loc=x_bar, scale=se)
    width = ci[1] - ci[0]
    print(f"信頼度{confidence*100:.0f}%: [{ci[0]:.4f}, {ci[1]:.4f}] "
          f"(幅: {width:.4f})")

# → 信頼度を上げると区間は広くなる（精度とのトレードオフ）