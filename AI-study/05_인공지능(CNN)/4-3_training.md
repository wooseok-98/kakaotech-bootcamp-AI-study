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

**딥러닝 모델의 손실 함수를 최소화하기 위해 기울기를 기반으로 가중치를 업데이트하는 알고리즘**

```
피드포워드 → 손실 계산 → 역전파(기울기 계산) → Optimizer(가중치 업데이트)
```

> 역전파는 **기울기를 계산**하는 과정, 실제 가중치를 **업데이트**하는 것은 Optimizer의 역할

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 손실 함수 최소화 | 손실 값이 작을수록 예측이 실제값에 가까움 — 옵티마이저가 이 방향으로 가중치 조정 |
| 가중치 조정 | 기울기를 기반으로 가중치를 업데이트하여 모델이 더 나은 예측을 하도록 학습 |
| 학습률 제어 | 학습률이 너무 크면 최적점을 지나치고, 너무 작으면 학습이 느림 — 옵티마이저가 적절히 제어 |
| 적응형 학습 | Adam 같은 적응형 옵티마이저는 파라미터마다 학습률을 자동으로 조정 |

### 옵티마이저 종류

**경사 하강법 계열 (Gradient Descent)**

| 종류 | 설명 | 권장 학습률 | 권장 상황 |
| --- | --- | --- | --- |
| 배치 경사 하강법 | 전체 데이터를 한 번에 사용해 가중치 업데이트 | 0.01~0.1 | 볼록 최적화, 계산 자원이 충분한 경우 |
| SGD | 하나의 샘플을 사용해 가중치 업데이트 | 0.001~0.01 | 데이터가 매우 크거나 연산량이 적은 환경 |
| 미니 배치 경사 하강법 | 소규모 배치 단위로 가중치 업데이트 | 0.001~0.01 | 일반적인 신경망 훈련 (CNN, NLP 등) |

**적응형 옵티마이저 (Adaptive Optimizers)**

| 종류 | 설명 | 권장 학습률 | 권장 상황 |
| --- | --- | --- | --- |
| Adagrad | 자주 업데이트된 파라미터의 학습률을 낮추고 드물게 업데이트된 것은 높임 | 0.01 | 희소 데이터, NLP, 추천 시스템 |
| RMSprop | 최근 기울기에 가중치를 두어 학습률 조정 | 0.001 | RNN/LSTM, 강화학습 |
| **Adam** | 모멘텀 + RMSprop의 장점을 결합 — 현재 가장 널리 사용 | 0.001 | 대부분의 신경망 (CNN, Transformer 등) |

**모멘텀 계열**

| 종류 | 설명 |
| --- | --- |
| NAG (Nesterov) | 모멘텀의 확장 버전 — 업데이트 전 기울기를 미리 계산해 더 정확한 업데이트 |

### 코드 예시

**TensorFlow**

```python
from tensorflow.keras.optimizers import SGD, Adam

# SGD 옵티마이저
optimizer = SGD(learning_rate=0.01)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Adam 옵티마이저 (기본 권장)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

**PyTorch**

```python
import torch.optim as optim

# SGD 옵티마이저
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Adam 옵티마이저 (기본 권장)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 공통 학습 루프
for inputs, labels in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(inputs), labels)
    loss.backward()
    optimizer.step()
```

---

## 경사 하강법 (Gradient Descent)

**손실 함수를 최소화하기 위해 기울기의 반대 방향으로 가중치를 반복적으로 조정하는 최적화 알고리즘**

기울기(gradient)를 계산하여 손실이 감소하는 방향으로 가중치를 조금씩 이동

### 동작 방식 (6단계)

```python
import numpy as np

# 1. 초기화
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
theta = np.random.randn(2, 1)          # 가중치 무작위 초기화

X_b = np.c_[np.ones((100, 1)), X]     # 2. 순전파: 절편 항(1) 추가

def compute_cost(X_b, y, theta):       # 3. 손실 계산 (MSE)
    m = len(y)
    return (1 / (2 * m)) * np.sum(np.square(X_b.dot(theta) - y))

def compute_gradient(X_b, y, theta):   # 4. 경사(기울기) 계산
    m = len(y)
    return 2 / m * X_b.T.dot(X_b.dot(theta) - y)

def gradient_descent(X_b, y, theta, lr, n_iter):   # 5. 가중치 업데이트 + 6. 반복
    for _ in range(n_iter):
        theta = theta - lr * compute_gradient(X_b, y, theta)
    return theta

theta_best = gradient_descent(X_b, y, theta, lr=0.1, n_iter=1000)
print("최적 파라미터:", theta_best)   # 약 [4.0, 3.0]
```

### Local Minima vs Global Minima

| 개념 | 설명 |
| --- | --- |
| Global Minima | 전체 손실 함수에서 가장 낮은 지점 — 최상의 최적점 |
| Local Minima | 특정 구간에서만 최소인 지점 — 전역 최적점이 아닐 수 있음 |

> **학습률(Learning Rate) 설정이 중요한 이유**
> - 너무 크면 → 최적점을 지나쳐 수렴 실패
> - 너무 작으면 → Local Minima에 갇혀 Global Minima 도달 불가
> - **Local Minima가 항상 나쁜 것은 아니며, Global Minima가 꼭 최상의 성능을 보장하지도 않음**
> - 고차원 공간에서는 안장점(Saddle Point)도 고려 필요

**기울기 소실 (Vanishing Gradient)**

역전파 과정에서 깊은 층으로 갈수록 기울기 값이 0에 수렴하여 가중치 업데이트가 거의 이루어지지 않는 현상
- 주로 Sigmoid·Tanh 활성화 함수에서 발생
- ReLU로 일부 완화 가능 (단, Dead Neuron 문제 주의)

---

## Adam (Adaptive Moment Estimation)

**학습률을 적응적으로 조정하고 1차·2차 모멘텀을 추정하여 학습을 가속화하고 안정화하는 딥러닝 최적화 알고리즘**

모멘텀(SGD 개선) + RMSprop(적응형 학습률)의 장점을 결합 — 현재 가장 널리 사용되는 옵티마이저

### 핵심 특성

| 항목 | 설명 |
| --- | --- |
| 1차 모멘텀 | 기울기의 이동 평균 — 업데이트 방향을 부드럽게 조정 |
| 2차 모멘텀 | 기울기 제곱의 이동 평균 — 파라미터마다 학습률을 개별 조정 |
| 적응적 학습률 | 파라미터마다 다른 학습률 적용 — 빠르고 안정적인 수렴 |
| 바이어스 수정 | 초기 모멘텀 추정치의 편향을 보정 — 초기 단계 안정화 |

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 빠른 수렴 | 기본 SGD보다 빠르게 최적점에 도달 |
| 노이즈 강인성 | 미니배치 학습 시 기울기 변화에도 안정적 |
| 초기 설정 민감도 낮음 | 기본 하이퍼파라미터(β₁=0.9, β₂=0.999, lr=0.001)로도 좋은 성능 |
| 범용성 | 이미지 분류, NLP, 강화학습 등 대부분의 신경망에 효과적 |

### vs SGD

| | SGD | Adam |
| --- | --- | --- |
| 학습률 | 고정 | 파라미터마다 자동 조정 |
| 수렴 속도 | 느림, 진동 큼 | 빠름, 안정적 |
| 하이퍼파라미터 민감도 | 높음 | 낮음 |

### 코드 예시

**TensorFlow**

```python
import tensorflow as tf

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,     # 1차 모멘텀 감소율
    beta_2=0.999,   # 2차 모멘텀 감소율
    epsilon=1e-07   # 수치 안정성을 위한 작은 상수
)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=10, batch_size=32)
```

**PyTorch**

```python
import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),  # (beta1, beta2)
    eps=1e-08,
    weight_decay=0        # L2 정규화 계수
)

for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
```

| 파라미터 | 설명 |
| --- | --- |
| `lr` / `learning_rate` | 학습률 — 기본값 0.001 |
| `beta_1` / `betas[0]` | 1차 모멘텀 감소율 — 기본값 0.9 |
| `beta_2` / `betas[1]` | 2차 모멘텀 감소율 — 기본값 0.999 |
| `epsilon` / `eps` | 수치 안정성을 위한 작은 상수 — 기본값 1e-7~1e-8 |
