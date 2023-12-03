import numpy as np
import matplotlib.pyplot as plt

# 예제 입력 데이터
X = np.array([1, 2, 3, 4, 5])
# 예제 출력 데이터
y = np.array([2, 4, 6, 8, 10])

# 초기 가중치 설정
w = 0
b = 0

# 학습률(learning rate) 설정
learning_rate = 0.01
# 학습 횟수(epoch) 설정
epochs = 1000

# 모델 학습
for epoch in range(epochs):
    # 예측값 계산
    y_pred = w * X + b
    # 손실(loss) 계산 (평균 제곱 오차)
    loss = np.mean((y_pred - y)**2)
    # 손실에 대한 가중치(w)와 절편(b)의 편미분 계산
    dw = (2/len(X)) * np.sum(X * (y_pred - y))
    db = (2/len(X)) * np.sum(y_pred - y)
    # 가중치와 절편 업데이트
    w = w - learning_rate * dw
    b = b - learning_rate * db
    # 매 100번째 에포크마다 손실 출력
    if epoch % 100 == 0:
        print(f'에포크 {epoch}, 손실: {loss:.4f}')

# 학습된 가중치와 절편 출력
print(f'학습된 가중치: {w:.2f}')
print(f'학습된 절편: {b:.2f}')