from scipy import stats
import numpy as np

# 例: 95%信頼区間の幅を±2以内にしたい
# 過去のデータから標準偏差は約10と推定
sigma_est = 10
E = 2  # 許容誤差
confidence = 0.95
z = stats.norm.ppf((1 + confidence) / 2)

n_required = (z * sigma_est / E) ** 2
print(f"=== 必要サンプルサイズの計算 ===")
print(f"推定標準偏差: {sigma_est}")
print(f"許容誤差: ±{E}")
print(f"信頼度: {confidence*100}%")
print(f"必要サンプルサイズ: {int(np.ceil(n_required))}以上")