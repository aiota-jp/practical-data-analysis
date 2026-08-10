from scipy import stats

# 二項分布を直接使った信頼区間（Clopper-Pearson法: 正確な方法）
n = 100
k = 65
confidence = 0.95

# Clopper-Pearson法（正確な二項信頼区間）
alpha = 1 - confidence
lower_cp = stats.beta.ppf(alpha / 2, k, n - k + 1)
upper_cp = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)

print(f"=== Clopper-Pearson法による正確な信頼区間 ===")
print(f"95%信頼区間: [{lower_cp:.4f}, {upper_cp:.4f}]")
print(f"           = [{lower_cp*100:.1f}%, {upper_cp*100:.1f}%]")