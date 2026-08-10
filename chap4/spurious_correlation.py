import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

np.random.seed(42)

# 第3の変数（気温）が2つの変数に影響を与える例
n = 50
temperature = np.random.normal(25, 8, n)  # 気温（交絡因子）

# 気温が高いとアイスが売れる
ice_cream_sales = 20 + 3 * temperature + np.random.normal(0, 10, n)

# 気温が高いと水難事故が増える
drowning_incidents = 1 + 0.3 * temperature + np.random.normal(0, 2, n)

# 相関係数の確認
from scipy import stats
r, p = stats.pearsonr(ice_cream_sales, drowning_incidents)
print(f"アイスクリーム売上と溺死事故の相関係数: {r:.4f}")
print(f"→ 高い正の相関があるが、因果関係ではない！")

# 可視化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# アイスvs溺死（擬似相関）
axes[0].scatter(ice_cream_sales, drowning_incidents, alpha=0.7, color='coral')
axes[0].set_xlabel("アイスクリーム売上")
axes[0].set_ylabel("水難事故件数")
axes[0].set_title(f"擬似相関 (R={r:.3f})\n※因果関係ではない", fontsize=11)

# 真の原因：気温vsアイス
r2, _ = stats.pearsonr(temperature, ice_cream_sales)
axes[1].scatter(temperature, ice_cream_sales, alpha=0.7, color='steelblue')
axes[1].set_xlabel("気温（℃）")
axes[1].set_ylabel("アイスクリーム売上")
axes[1].set_title(f"真の関係 (R={r2:.3f})\n気温→アイス売上", fontsize=11)

# 真の原因：気温vs溺死
r3, _ = stats.pearsonr(temperature, drowning_incidents)
axes[2].scatter(temperature, drowning_incidents, alpha=0.7, color='green')
axes[2].set_xlabel("気温（℃）")
axes[2].set_ylabel("水難事故件数")
axes[2].set_title(f"真の関係 (R={r3:.3f})\n気温→水難事故", fontsize=11)

plt.tight_layout()
plt.show()