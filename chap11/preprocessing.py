from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Irisデータの準備
iris = load_iris()
X = iris.data

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=== 標準化前 ===")
print(X[:5])

print("\n=== 標準化後 ===")
print(X_scaled[:5])