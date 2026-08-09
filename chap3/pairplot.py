import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns

# Irisデータセットを使った例
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue="species", diag_kind="hist")
plt.suptitle("Irisデータセットのペアプロット", y=1.02, fontsize=14)
plt.show()
