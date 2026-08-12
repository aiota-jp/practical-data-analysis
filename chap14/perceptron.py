import numpy as np

def perceptron(inputs, weights, bias):
    """単純パーセプトロン"""
    weighted_sum = np.dot(inputs, weights) + bias
    return 1 if weighted_sum > 0 else 0

# チョコレートを買うかの判断
# 甘いもの好きな人の重み設定
weights = np.array([0.3, 0.2, 0.8])  # 空腹, 懐事情, 甘いもの好き
bias = -0.5

# ケース1: 空腹で余裕あり、甘いもの好き
inputs1 = np.array([1, 1, 1])
print(f"ケース1(空腹,余裕あり,甘党): {perceptron(inputs1, weights, bias)} → 買う")

# ケース2: 満腹で余裕なし、甘いもの好き
inputs2 = np.array([0, 0, 1])
print(f"ケース2(満腹,余裕なし,甘党): {perceptron(inputs2, weights, bias)} → {'買う' if perceptron(inputs2, weights, bias) else '買わない'}")

# ケース3: 空腹だが余裕なし、甘いもの好きでもない
inputs3 = np.array([1, 0, 0])
print(f"ケース3(空腹,余裕なし,普通): {perceptron(inputs3, weights, bias)} → {'買う' if perceptron(inputs3, weights, bias) else '買わない'}")