# 4-3. Loss Function · Backpropagation · Optimizer

---

## 손실 함수 (Loss Function)

**인공신경망이나 기타 머신러닝 모델에서 예측값과 실제 값 간의 차이를 정량적으로 측정하는 함수**

- 손실 값이 클수록 모델의 예측 성능이 낮음을 의미
- 손실 함수는 **오차를 측정하는 역할만** 함 — 실제로 값을 조정하는 것은 Optimizer

### 손실 함수의 종류

**1. 회귀(Regression) 문제**

| 손실 함수 | 설명 | 특징 |
| --- | --- | --- |
| MSE (평균 제곱 오차) | 예측값과 실제값의 차이를 제곱한 후 평균 | 큰 오차일수록 더 크게 반영 — 이상치에 민감 |
| MAE (평균 절대 오차) | 예측값과 실제값 차이의 절댓값을 평균 | 이상치 영향이 적고, 원래 단위 그대로라 해석이 쉬움 |

| 비교 항목 | MSE | MAE |
| --- | --- | --- |
| 이상치 영향 | 큼 (오차 제곱) | 작음 (절댓값) |
| 미분 가능성 | 연속적으로 미분 가능 | 0에서 미분 불연속 |
| 해석 용이성 | 단위가 원래 데이터의 제곱 — 직관적 해석 어려움 | 원래 단위 그대로 — 해석 쉬움 |
| 활용 | 이상치를 강조할 때 (예: 금융 리스크) | 전반적 평균 오차를 볼 때 |

**2. 분류(Classification) 문제**

| 손실 함수 | 설명 | 주요 사용처 |
| --- | --- | --- |
| Cross-Entropy Loss | 확률 분포 간 차이를 측정 — 정답 확률이 높을수록 손실 감소 | 딥러닝 분류 (신경망 + softmax) |
| Binary Cross-Entropy | 이진 분류에서 예측 확률과 실제 레이블(0/1) 간의 차이 측정 | 스팸 필터링, 이진 분류 |
| Hinge Loss | 정답과 예측값 사이의 마진 기반 손실 — 마진을 넘으면 손실 0 | SVM 기반 분류 |

> **회귀**: 과거 데이터를 바탕으로 미래 값을 예측하거나 두 변수 간의 관계를 이해하는 방법 (예: 집값 예측, 매출 예측)

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 모델 성능 평가 | 예측값과 실제값 간의 차이를 정량적으로 평가 — 손실 값이 낮을수록 정확한 예측 |
| 가중치 조정 지표 | 경사 하강법(Gradient Descent) 등의 최적화 알고리즘이 손실을 줄이는 방향으로 가중치를 조정하는 기준 |
| 모델 개선 | 손실 값 분석을 통해 모델의 예측 오류를 파악하고 개선 |

### 코드 예시 (5단계)

```python
import numpy as np

# 1. 예측값 계산 (초기 가중치/편향으로 선형 모델)
x = np.array([1, 2, 3, 4, 5])
y_actual = np.array([2.5, 4.5, 6.5, 8.5, 10.5])
w, b = 0.5, 0.5

y_pred = w * x + b
print("예측값:", y_pred)  # [1.0, 1.5, 2.0, 2.5, 3.0]

# 2. 실제값과 비교
print("실제값:", y_actual)

# 3. 오차 계산 (MSE)
mse = np.mean((y_pred - y_actual) ** 2)
print("MSE:", mse)  # 24.75

# 4. 가중치 업데이트 (경사 하강법)
learning_rate = 0.01
for epoch in range(100):
    y_pred  = w * x + b
    mse     = np.mean((y_pred - y_actual) ** 2)
    grad_w  = 2 * np.mean((y_pred - y_actual) * x)  # MSE의 w에 대한 미분
    grad_b  = 2 * np.mean(y_pred - y_actual)          # MSE의 b에 대한 미분
    w      -= learning_rate * grad_w
    b      -= learning_rate * grad_b
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {mse:.4f}")

# 5. 반복 후 결과
print(f"최종 가중치: {w:.4f}, 최종 편향: {b:.4f}")
```

```
Epoch 0:  Loss = 24.7500
Epoch 10: Loss = 0.1370
Epoch 20: Loss = 0.0242
...
최종 가중치: 1.9238, 최종 편향: 0.7750
```

### 손실 함수 vs 비용 함수

| 구분 | 손실 함수 (Loss Function) | 비용 함수 (Cost Function) |
| --- | --- | --- |
| 정의 | 하나의 데이터 샘플에서 발생한 오차 측정 | 모든 데이터 샘플의 평균 오차 측정 |
| 목적 | 개별 예측의 정확도 평가 | 모델 전체 성능 평가 및 학습 기준 |
| 비유 | 시험 문제 하나당 감점 | 시험 전체 평균 점수 |

> **손실 값이 낮다고 항상 좋은 모델은 아님**
> - 과적합(Overfitting): 훈련 데이터 손실은 낮지만 테스트 데이터 성능은 저하
> - 불균형 데이터: 손실이 줄어도 정확도가 개선되지 않을 수 있음
> - 손실 감소는 성능 향상 가능성을 높이지만, 반드시 비례하지는 않음

---

## Backpropagation (역전파)

**Neural Network에서 출력값과 실제 값 간의 오차를 기반으로 각 뉴런의 가중치를 조정하기 위해 사용하는 알고리즘**

```
피드포워드 (Feed-Forward)
    → 손실 함수 (Loss Function) 계산
        → 역전파 (Backpropagation) — 기울기 계산
            → Optimizer — 가중치 업데이트
```

> **중요한 오해 정정**
> - Backpropagation은 **기울기(Gradient)를 계산**하는 과정
> - 실제 가중치를 조정하는 것은 **Optimizer의 역할**
> - 기울기는 임의로 조정하는 값이 아니라, 가중치를 조정하면 자연스럽게 변하는 값

### 체인 룰 (Chain Rule)

신경망은 여러 층으로 구성되어 있어, 각 층의 가중치가 최종 손실에 미치는 영향을 직접 계산하기 어려움
→ **연쇄 법칙(Chain Rule)** 으로 해결: 출력층 → 은닉층 → 입력층 방향으로 미분값을 단계적으로 전파

**도미노 효과로 이해**:
1. 출력층의 손실(마지막 도미노)이 발생
2. 그 충격이 은닉층으로 전달
3. 은닉층 → 입력층까지 차례로 영향이 전달
4. 이 과정에서 각 가중치가 손실에 미치는 영향을 계산

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 빠르고 정확한 학습 | 각 가중치의 기울기를 효율적으로 계산하여 모델이 점차 더 정확한 예측을 하도록 학습 |
| 최적화 문제 해결 | 손실 함수를 최소화하는 가중치 조합을 경사 하강법으로 탐색 |
| 복잡한 모델 학습 | 다층 신경망에서 각 층의 가중치를 정확하게 조정 가능 — 고수준 추상화 학습 지원 |

### 코드 예시

**TensorFlow**

```python
import tensorflow as tf
import numpy as np

# AND 게이트 데이터
inputs = tf.constant([[0,0],[0,1],[1,0],[1,1]], dtype=tf.float32)
labels = tf.constant([[0],[0],[0],[1]], dtype=tf.float32)

# 모델 정의
class SimpleNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = tf.keras.layers.Dense(2, activation='relu')
        self.out    = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        return self.out(self.hidden(x))

model = SimpleNN()
criterion = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# 역전파 학습 루프
for epoch in range(10000):
    with tf.GradientTape() as tape:          # 순전파 구간 기록
        outputs = model(inputs)
        loss    = criterion(labels, outputs)

    grads = tape.gradient(loss, model.trainable_variables)  # 기울기 계산
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 가중치 업데이트

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss.numpy():.4f}")

# 예측
binary_out = np.where(outputs.numpy() > 0.5, 1, 0)
print("예측 결과:", binary_out.flatten())
```

| 코드 | 설명 |
| --- | --- |
| `tf.GradientTape()` | 순전파 연산을 기록하여 나중에 기울기를 계산할 수 있게 하는 컨텍스트 |
| `tape.gradient(loss, variables)` | 각 변수에 대한 손실의 기울기를 역전파로 계산 |
| `optimizer.apply_gradients(...)` | 계산된 기울기를 사용해 가중치 업데이트 |

**PyTorch**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# AND 게이트 데이터
inputs = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
labels = torch.tensor([[0],[0],[0],[1]], dtype=torch.float32)

# 모델 정의
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden  = nn.Linear(2, 2)
        self.output  = nn.Linear(2, 1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        return self.sigmoid(self.output(x))

model     = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 역전파 학습 루프
for epoch in range(10000):
    outputs = model(inputs)                # 순전파
    loss    = criterion(outputs, labels)   # 손실 계산

    optimizer.zero_grad()  # 이전 기울기 초기화
    loss.backward()        # 역전파 — 각 파라미터의 기울기 계산
    optimizer.step()       # 기울기를 바탕으로 가중치 업데이트

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 예측
with torch.no_grad():
    print("예측 결과:", (outputs > 0.5).int().flatten().tolist())
```

| 코드 | 설명 |
| --- | --- |
| `optimizer.zero_grad()` | 이전 에폭의 기울기를 0으로 초기화 — 누적 방지 |
| `loss.backward()` | 손실에 대한 각 파라미터의 기울기를 역전파로 계산 |
| `optimizer.step()` | 계산된 기울기를 이용해 가중치 업데이트 |
| `BinaryCrossentropy / BCELoss` | 이진 분류 손실 함수 — 예측 확률과 실제 라벨 간의 차이 측정 |
| `SGD` | 확률적 경사 하강법 — 미니배치 단위로 가중치를 업데이트 |

---

## Optimizer (옵티마이저)

> 다음 챕터 내용 수신 후 추가 예정
