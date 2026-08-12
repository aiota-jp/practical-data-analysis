from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

data = np.array([[100, 50000], [200, 80000], [150, 60000]])

# Min-Max法
mm_scaler = MinMaxScaler()
data_mm = mm_scaler.fit_transform(data)

# Zスコア標準化
std_scaler = StandardScaler()
data_std = std_scaler.fit_transform(data)

print("=== スケーリング比較 ===")
print(f"元データ:\n{data}")
print(f"\nMin-Max法:\n{data_mm}")
print(f"\nZスコア標準化:\n{data_std}")