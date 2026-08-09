from scipy import stats

data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
mode_val = stats.mode(data, keepdims=True)
print(f"最頻値: {mode_val.mode[0]}")  # 3（3回出現）
