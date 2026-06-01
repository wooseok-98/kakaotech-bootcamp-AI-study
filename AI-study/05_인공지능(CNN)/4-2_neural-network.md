# 4-2. Activation Function · ANN · FCNN · FCL

---

## 비선형 활성화 함수 (Activation Function)

**인공신경망에서 뉴런의 출력을 결정하는 비선형 함수**

가중합을 계산한 후 이를 비선형 함수에 적용하여 최종 출력을 생성
- **선형 함수**: 전체 구간에서 동일한 기울기를 가지는 직선
- **비선형 함수**: 전체 구간에서 동일한 기울기를 갖지 않는 함수 (곡선 형태)

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 비선형성 도입 | 선형으로 분리할 수 없는 문제(XOR 등) 해결 가능 |
| 학습 가능성 향상 | 다양한 데이터 패턴을 더 잘 학습할 수 있게 함 |
| 다층 신경망 구현 | 각 층에서 비선형 변환을 적용해 고차원 패턴 학습 가능 |

> **XOR이 비선형 문제인 이유**: `(0,0), (1,1) → 0` / `(0,1), (1,0) → 1` 로, 두 그룹을 직선 하나로 나눌 수 없음

### 1. 시그모이드 (Sigmoid)

모든 입력 값을 **0과 1 사이**로 매핑하는 S자 형태의 함수

```
σ(x) = 1 / (1 + e^(-x))
```

| x | e^(-x) | 결과 |
| --- | --- | --- |
| -5 | 148.4 | ≈ 0.007 |
| 0 | 1 | 0.5 |
| +5 | 0.007 | ≈ 0.993 |

| 장점 | 단점 |
| --- | --- |
| 출력이 0~1 사이 — 확률값 표현에 적합 | 기울기 소실(Vanishing Gradient) 문제 발생 가능 |
| | 출력이 0 또는 1에 가까워질수록 기울기가 0에 수렴 → 역전파 시 가중치 업데이트가 거의 안 됨 |

### 2. 하이퍼볼릭 탄젠트 (tanh)

모든 입력 값을 **-1과 1 사이**로 매핑하는 S자 형태의 함수

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

| x | tanh 결과 |
| --- | --- |
| -5 | ≈ -0.9999 |
| 0 | 0 |
| +5 | ≈ +0.9999 |

| 장점 | 단점 |
| --- | --- |
| 중심이 0 — 시그모이드보다 학습이 안정적 | 기울기 소실 문제 여전히 발생 가능 |
| 대칭성이 있어 학습이 더 빠를 수 있음 | 출력이 -1 또는 1에 가까워질수록 기울기 소멸 |

> 시그모이드·tanh·계단함수 차이
> - 시그모이드: 연속 곡선, 출력 (0, 1)
> - tanh: 연속 곡선, 출력 (-1, 1), 중심 0
> - 계단 함수: 비연속, 미분 불가 — 딥러닝 학습에 부적합

### 3. 렐루 (ReLU, Rectified Linear Unit)

입력이 0 이하이면 0, 0 초과이면 그대로 출력하는 함수

```
f(x) = max(0, x)
```

| x | 결과 |
| --- | --- |
| -5 | 0 |
| 0 | 0 |
| +3 | 3 |
| +10 | 10 |

| 장점 | 단점 |
| --- | --- |
| 계산이 단순하고 빠름 | 음수 입력에서 뉴런이 죽는(Dead Neuron) 문제 발생 가능 |
| 기울기 소실 문제 완화 — 양수 구간에서 기울기가 1로 일정 | 입력이 음수이면 기울기 0 → 해당 뉴런은 더 이상 학습 안 됨 |
| 현재 딥러닝의 기본 활성화 함수 | |

> ReLU가 비선형인 이유: x < 0 구간(기울기 0)과 x > 0 구간(기울기 1)의 특성이 달라서 전체적으로 비선형 함수로 분류됨

### 코드 예시

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# 가중합 계산
input_value = 0.5
weight = 0.8
bias = 0.1
weighted_sum = input_value * weight + bias

print(f"가중합: {weighted_sum}")
print(f"Sigmoid: {sigmoid(weighted_sum):.4f}")
print(f"Tanh:    {tanh(weighted_sum):.4f}")
print(f"ReLU:    {relu(weighted_sum):.4f}")

# 다음 층에 전달
next_layer_input = sigmoid(weighted_sum)
```

---

## ANN (Artificial Neural Network, 인공신경망)

**머신러닝과 인지 과학에서 사용되어 패턴 인식과 문제 해결 능력을 갖추게 하는, 뇌의 뉴런 네트워크를 모방한 알고리즘**

### 구조

```
입력층 (Input Layer) → 은닉층 (Hidden Layer) → 출력층 (Output Layer)
```

| 구성 요소 | 설명 |
| --- | --- |
| 입력층 (Input Layer) | 입력 데이터를 받음 — 주로 1차원 배열(Flatten) 형태 |
| 은닉층 (Hidden Layer) | 입력값에 가중치를 곱하고 활성화 함수를 적용해 특징 추출 |
| 출력층 (Output Layer) | 최종 예측값 출력 |

층과 층 사이에는 **가중치(Weight) 행렬**이 존재하며, 이 가중치가 학습의 핵심

### 학습 전체 과정

```
1. 피드포워드 (Feed-Forward)
   입력 → 은닉층(가중치×활성화 함수) → 출력

2. 손실 함수 (Loss Function)
   예측값 vs 실제 정답 → 손실(Loss) 계산

3. 역전파 (Backpropagation)
   손실을 줄이기 위해 각 가중치의 기울기(gradient) 계산 (출력층 → 입력층 방향)

4. 경사 하강법 (Gradient Descent)
   기울기를 이용해 가중치 업데이트 → 손실 최소화
```

> **Feed-Forward(순전파)**: 입력 데이터를 출력으로 변환하는 과정만 의미 — 오차 계산은 포함하지 않음
> **Backpropagation(역전파)**: 손실을 기반으로 역방향으로 기울기(gradient)를 계산하는 과정

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 자동 학습 | 인간이 일일이 개입하지 않아도 데이터에서 패턴을 스스로 학습 |
| 복잡한 문제 해결 | 규칙 기반 프로그래밍으로 해결하기 어려운 문제 (컴퓨터 비전, 음성 인식 등) 해결 가능 |
| 범용성 | 이미지, 음성, 텍스트 등 다양한 데이터 형식에 적용 가능 |

### XOR 예제 코드 (사용 방법)

```python
import numpy as np

# 데이터
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 초기화
np.random.seed(42)
weights = np.random.rand(2, 2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

learning_rate = 0.1

for _ in range(1000):
    # 1. 피드포워드
    weighted_inputs = np.dot(X, weights)
    output = sigmoid(weighted_inputs)

    # 2. 오차 계산
    error = y - output

    # 3. 역전파 (시그모이드 미분: output * (1 - output))
    adjustments = learning_rate * np.dot(X.T, error * (output * (1 - output)))

    # 4. 가중치 업데이트
    weights += adjustments

print("최종 가중치:", weights)
print("예측값:", output)
```

| 코드 | 설명 |
| --- | --- |
| `np.dot(X, weights)` | 입력과 가중치의 행렬 곱 — 가중합 계산 |
| `error * (output * (1 - output))` | 시그모이드의 미분값을 반영한 기울기 계산 (역전파 핵심) |
| `X.T` | 입력 행렬의 전치(행↔열 교환) — 행렬 곱셈 차원을 맞추기 위해 사용 |
| `weights += adjustments` | 경사 하강법으로 가중치 업데이트 |

---

## FCNN (Fully Connected Neural Network, 완전 연결 신경망)

**모든 뉴런이 이전 층의 모든 뉴런과 연결된 신경망 구조**

| ANN vs FCNN |  |
| --- | --- |
| ANN | 다양한 신경망 구조를 포함하는 포괄적 용어 — 모든 뉴런을 연결하지 않아도 됨 |
| FCNN | ANN의 한 종류 — 반드시 모든 뉴런이 완전히 연결된 구조 |

> 모든 FCNN은 ANN이지만, 모든 ANN이 FCNN은 아님

### 장단점

| 장점 | 단점 |
| --- | --- |
| 구현이 간단하고 이해하기 쉬움 | 특징 수가 많아지면 연산량 급격히 증가 |
| 모든 특징을 종합적으로 학습 — 복잡한 패턴 인식 | 데이터가 많아질수록 과적합 위험 |
| 분류·회귀 등 다양한 예측 작업에 사용 가능 | 이미지·시계열 등 특수 데이터에는 성능이 떨어질 수 있음 |

### 코드 예시

**TensorFlow**

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# 데이터 준비
input_dim, num_classes = 20, 10
X_train = np.random.rand(1000, input_dim).astype(np.float32)
y_train = np.random.randint(num_classes, size=1000).astype(np.int64)
X_test  = np.random.rand(200, input_dim).astype(np.float32)
y_test  = np.random.randint(num_classes, size=200).astype(np.int64)

# 모델 정의
model = Sequential([
    Input(shape=(input_dim,)),
    Dense(64, activation='relu'),        # 은닉층 1
    Dense(64, activation='relu'),        # 은닉층 2
    Dense(num_classes, activation='softmax')  # 출력층 — 다중 분류
])

# 컴파일 및 학습
model.compile(optimizer=Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32)

# 평가
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")
```

| 개념 | 설명 |
| --- | --- |
| `softmax` | 각 클래스의 확률을 계산, 합이 1 — 다중 분류의 출력층에 사용 |
| `SparseCategoricalCrossentropy` | 정답 라벨이 정수 인덱스일 때 사용하는 다중 분류 손실 함수 |
| 로짓 (logit) | 활성화 함수 적용 전의 원본 점수 — `from_logits=True`로 설정 |

**PyTorch**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 데이터 준비
input_dim, num_classes = 20, 10
X_train = torch.tensor(np.random.rand(1000, input_dim), dtype=torch.float32)
y_train = torch.tensor(np.random.randint(num_classes, size=1000), dtype=torch.long)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# 모델 정의
class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 은닉층 1
        x = torch.relu(self.fc2(x))  # 은닉층 2
        return self.fc3(x)           # 출력층

model = FCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
for epoch in range(20):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

| `CrossEntropyLoss` | 정수 라벨 기반 다중 분류 손실 함수 — softmax + log-likelihood 결합 |
| --- | --- |
| `nn.Linear(in, out)` | PyTorch에서의 Fully Connected Layer — 선형 변환 수행 |

### 다른 신경망과 비교

| 신경망 | 특화 분야 | 주요 특징 |
| --- | --- | --- |
| **FCNN** | 범용 분류·회귀 | 모든 뉴런 완전 연결, 구현 단순 |
| **CNN** | 이미지 처리 | 합성곱 연산으로 지역 패턴 학습, 파라미터 공유 |
| **RNN** | 순차 데이터 | 이전 시점 상태를 현재에 반영, 직렬 학습 |
| **LSTM** | 장기 의존성 | RNN의 기울기 소실 문제 개선, 중요 정보 유지·불필요 정보 망각 |

---

## FCL (Fully Connected Layer, 완전 연결 계층)

**ANN에서 모든 입력 뉴런이 모든 출력 뉴런과 연결된 레이어**

- Hidden Layer와 Output Layer가 FCL에 해당
- TensorFlow: **Dense Layer** / PyTorch: **Linear Layer** 라고도 불림

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 복잡한 패턴 학습 | 모든 특징 간의 상호작용을 학습해 복잡한 패턴 인식 |
| 최종 출력 생성 | 추출된 모든 특징을 종합하여 최종 분류·회귀 결과 도출 |
| 다양한 적용 분야 | 이미지·음성·자연어 처리 등 광범위하게 활용 |

> FCL은 자주 사용되지만 **필수 구조는 아님** — 모델 목적과 구조에 따라 선택적으로 사용

### 코드 예시

**TensorFlow**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# 데이터 준비
X = np.random.rand(1000, 20)
y = np.random.randint(2, size=1000)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 모델 — FCL(Dense) 구성
model = Sequential([
    Input(shape=(20,)),
    Dense(64, activation='relu'),   # FCL 1
    Dense(32, activation='relu'),   # FCL 2
    Dense(1,  activation='sigmoid') # FCL 출력 (이진 분류)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
```

**PyTorch**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 이미지 형태 데이터 (32×32×3)
x_train = torch.tensor(np.random.rand(1000, 3, 32, 32), dtype=torch.float32)
y_train = torch.tensor(np.random.randint(10, size=1000), dtype=torch.long)
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)

# FCL 포함 모델 — Flatten 후 Linear(FCL) 적용
class SimpleFCL(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 256)  # FCL 1
        self.fc2 = nn.Linear(256, 10)             # FCL 출력

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleFCL()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

### FCL의 한계

**파라미터 수 급증** — 입력 1,000개 × 출력 500개 = 500,000개 가중치 + 500개 편향 = 500,500개

| 한계 | 설명 |
| --- | --- |
| 파라미터 수 폭발 | 층이 깊어지면 수억 개의 가중치가 필요 → 메모리·연산 비용 급증 |
| 과적합 위험 | 파라미터가 많아질수록 훈련 데이터를 암기하는 경향 — 일반화 성능 저하 |
| 불필요한 학습 | 중요하지 않은 특징까지 모두 학습 → 비효율적 |

> FCL의 과적합 해결 방법: Dropout, 데이터 증강, 정규화(L1/L2), 더 많은 학습 데이터 확보
