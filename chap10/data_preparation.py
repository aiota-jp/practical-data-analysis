import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import japanize_matplotlib

# 乳がんデータセット（2クラス分類用）
cancer = load_breast_cancer()
X_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_cancer = cancer.target  # 0: 悪性, 1: 良性

print("=== 乳がんデータセット ===")
print(f"データ数: {X_cancer.shape[0]}, 特徴量数: {X_cancer.shape[1]}")
print(f"クラス: {cancer.target_names}")
print(f"クラスごとの件数: 悪性={sum(y_cancer==0)}, 良性={sum(y_cancer==1)}")

# 学習データとテストデータに分割
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer)

# Irisデータセット（多クラス分類用）
iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = iris.target  # 0: setosa, 1: versicolor, 2: virginica

print(f"\n=== Irisデータセット ===")
print(f"データ数: {X_iris.shape[0]}, 特徴量数: {X_iris.shape[1]}")
print(f"クラス: {iris.target_names}")

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris)
