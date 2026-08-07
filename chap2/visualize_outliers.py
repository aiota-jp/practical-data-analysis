import pandas as pd
import matplotlib.pyplot as plt

# サンプルデータ
df = pd.DataFrame({
    "value": [10, 12, 14, 15, 13, 11, 30, 14, 12, 13, -10]
})

print("===== 元のデータ =====")
print(df)


# ========================================
# 箱ひげ図
# ========================================

plt.figure(figsize=(6, 3))

plt.boxplot(
    df["value"],
    # vert=False
)

plt.xlabel("Value")
plt.title("Boxplot")

plt.show()