# 4-1. Deep Learning · Perceptron

---

## Deep Learning (딥 러닝)

**데이터를 스스로 학습하여 예측하거나 분류하는 모델을 자동으로 만들어내는 기술**

- **비선형** + **자동화**가 핵심
- 사람이 직접 규칙을 정하지 않아도 비선형적인 패턴(복잡한 관계)을 스스로 찾아냄
- 머신러닝(ML)의 하위 분야

> **비선형**: 입력과 출력 사이의 관계가 직선으로 표현되지 않는 특성 — 입력 변화에 따라 출력이 곡선 형태로 변화

### 딥러닝 동작 흐름

```
데이터 준비 → 모델 설계 → 모델 초기화 → 훈련 → 평가
    → 성능 만족? → Yes: 배포
                 ↓ No
            하이퍼파라미터 조정 → 훈련 ↩
```

| 단계 | 이름 | 설명 |
| --- | --- | --- |
| 1 | 데이터 준비 | 딥러닝 모델 훈련을 위한 데이터 준비 |
| 2 | 모델 설계 | 신경망 구조(층 수, 뉴런 수, 활성화 함수) 설계 |
| 3 | 모델 초기화 | 가중치(weight)와 편향(bias)을 초기화 |
| 4 | 훈련 | 훈련 데이터로 모델 학습, 손실 함수와 옵티마이저로 가중치 조정 |
| 5 | 평가 | 테스트 데이터로 성능 평가 |
| 6 | 하이퍼파라미터 조정 | 성능 미달 시 설정값 수정 후 재훈련 |
| 7 | 배포 | 성능 만족 시 실제 환경에 배포 |

### 딥러닝 vs 머신러닝

| 비교 항목 | 머신러닝 | 딥러닝 |
| --- | --- | --- |
| 특징 추출 | 수동 (사람이 직접) | 자동 |
| 데이터 요구량 | 상대적으로 적음 | 매우 많음 |
| 계산 복잡도 | 낮음 | 높음 |
| 성능 | 단순 문제에 강점 | 복잡한 문제에 강점 |
| 유연성 | 특정 문제에 특화 | 다양한 문제에 적용 가능 |

> **딥러닝 vs 머신러닝 플로우 차이**
> - 딥러닝: 모델 설계(신경망 구조) + 모델 초기화(가중치·편향 초기화) 단계 추가
> - 머신러닝: 모델 선택 및 훈련, 하이퍼파라미터 조정 단계 중심

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 높은 예측 정확도 | ANN 구조를 통해 복잡한 데이터에서 높은 수준의 특징을 자동 학습 — 이미지·음성·자연어 처리에서 뛰어난 성능 발휘 |
| 자동화된 특징 학습 | 전통 ML과 달리 데이터에서 유용한 특징을 직접 추출할 필요 없이 자동 학습 |
| 확장성 및 유연성 | 매우 큰 규모의 데이터 처리 가능, 다양한 응용 분야에 적용 가능 |

### 핵심 개념

**Bias (편향)**

`y = Wx + b` 에서 b — 입력 데이터에 추가되는 상수 값

- 결정 경계를 y축 방향으로 이동시켜 모델의 표현력을 높임
- Bias가 없으면 결정 경계가 원점을 반드시 지나야 함 → 표현력 제한
- Bias가 있으면 결정 경계를 자유롭게 평행 이동 가능 → 더 유연한 분류·회귀

**Hyperparameter (하이퍼파라미터)**

개발자가 학습 시작 전 미리 설정해야 하는 변수 — 학습 과정 중 고정

| 하이퍼파라미터 | 설명 | 영향 |
| --- | --- | --- |
| 학습률 (Learning Rate) | 가중치 업데이트 크기 결정 | 너무 높으면 수렴 불안정, 너무 낮으면 수렴 느림 |
| 배치 크기 (Batch Size) | 한 번의 학습 단계에서 사용하는 샘플 수 | 크면 안정적이지만 메모리 증가, 작으면 메모리 절약하지만 불안정 |
| 에폭 (Epoch) | 전체 데이터셋을 학습하는 반복 횟수 | 많으면 더 많이 학습하지만 과적합 위험 |
| 층 수 및 뉴런 수 | 신경망 구조 결정 | 복잡할수록 다양한 패턴 학습 가능하지만 계산 비용 증가·과적합 위험 |
| 드롭아웃 비율 | 학습 중 무작위로 비활성화하는 뉴런 비율 | 과적합 방지, 일반화 성능 향상 |

### 사용 방법 (5단계)

| 단계 | 설명 |
| --- | --- |
| 데이터 준비 | 데이터 수집 → 정제 → 전처리 (크기 조정, 정규화 등) |
| 신경망 모델 설계 | 입력층·은닉층·출력층 구성, 각 층의 뉴런 수·활성화 함수 설계 |
| 모델 학습 | 손실 함수와 최적화 알고리즘으로 가중치 조정, 여러 에폭 반복 |
| 모델 평가 | 검증 데이터로 성능 평가, 필요시 모델 개선 |
| 모델 최적화 | 하이퍼파라미터 튜닝, 재학습, 모델 구조 변경 등 |

### XOR 예제 코드

XOR(배타적 OR): 두 입력이 서로 다를 때만 1, 같으면 0

| 입력 A | 입력 B | A XOR B |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**TensorFlow**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# XOR 데이터
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 모델 설계
model = Sequential([
    Input(shape=(2,)),
    Dense(2, activation='relu'),      # 은닉층
    Dense(1, activation='sigmoid')    # 출력층 — 이진 분류
])

# 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습
model.fit(X, y, epochs=1000, verbose=0)

# 평가 및 예측
loss, accuracy = model.evaluate(X, y)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

predictions = model.predict(X)
print(f'Predictions: {predictions}')
```

| 파라미터 | 설명 |
| --- | --- |
| `binary_crossentropy` | 이진 분류용 손실 함수 — 두 클래스(0/1) 중 하나로 분류할 때 사용 |
| `adam` | 학습률을 자동 조정하는 옵티마이저 — 모멘텀과 제곱 평균을 활용 |
| `verbose=0` | 학습 과정 로그 미출력 |

**PyTorch**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# XOR 데이터
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 모델 설계
class XORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

model = XORModel()

# 손실 함수 & 옵티마이저
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# 평가
model.eval()
with torch.no_grad():
    outputs   = model(X)
    predicted = (outputs > 0.5).float()
    accuracy  = (predicted == y).float().mean()
    print(f'Loss: {criterion(outputs, y).item():.4f}, Accuracy: {accuracy.item():.4f}')
    print(f'Predictions: {outputs}')
```

| 코드 | 설명 |
| --- | --- |
| `optimizer.zero_grad()` | 이전 배치의 기울기를 초기화 — 누적 방지 |
| `loss.backward()` | 역전파로 각 파라미터의 기울기 계산 |
| `optimizer.step()` | 계산된 기울기로 가중치 업데이트 |
| `torch.no_grad()` | 평가 시 기울기 계산 비활성화 — 메모리 절약 |

---

## Perceptron (퍼셉트론)

**ANN(Artificial Neural Network)의 기본 단위 — 입력값을 가중치(weight)와 함께 처리하여 단일 출력을 생성하는 선형 이진 분류기**

사람의 뉴런(dendrites → cell body → axon)이 여러 신호를 받아 하나의 결과를 전달하는 방식을 수학적으로 단순화한 모델

### 퍼셉트론의 구조

```
입력(x1, x2, ...)
     ↓ 각 입력 × 가중치(weight)
가중합(Weighted Sum): Σ(xᵢ × wᵢ) + bias
     ↓
활성화 함수(Activation Function)
     ↓
출력(0 또는 1)
```

| 구성 요소 | 설명 |
| --- | --- |
| 가중합 (Weighted Sum) | 각 입력과 가중치를 곱한 후 모두 더하고, 편향(bias)을 더한 값 |
| 활성화 함수 | 가중합을 입력받아 최종 출력값 결정 |

### 단층 퍼셉트론 vs 다층 퍼셉트론

| 항목 | 단층 퍼셉트론 | 다층 퍼셉트론 (MLP) |
| --- | --- | --- |
| 활성화 함수 | 계단 함수 (Step Function) — 임계값 이상이면 1, 아니면 0 | ReLU, Sigmoid, Tanh 등 비선형 활성화 함수 |
| 결정 경계 | 선형 (직선/초평면) | 비선형 — 복잡한 패턴 학습 가능 |
| 해결 가능 문제 | AND, OR 등 선형 분리 가능한 문제 | XOR 등 비선형 문제 포함 |

> 단층 퍼셉트론에 비선형 활성화 함수를 넣어도 가중치 결합 자체가 선형이기 때문에 선형 결정 경계만 형성됨
> → **비선형 문제를 해결하려면 다층 구조(MLP)가 필요**

### Weight (가중치)

입력 값이 결과에 얼마나 **영향**을 미치는지 조절하는 값

- 모델이 각 입력의 중요도를 파악하여 더 적절한 예측을 할 수 있도록 도움
- 학습 과정에서 오차에 따라 자동으로 조정됨

### Bias (편향)

가중치와 별개로 결과에 추가적으로 영향을 줄 수 있는 값

- 모든 입력이 0일 때도 모델이 올바른 출력을 할 수 있게 도움
- 결정 경계를 평행 이동시켜 모델 표현력 향상

> 가중치와 편향을 초기화할 때 랜덤값을 사용하는 이유: **대칭 깨기(symmetry breaking)**
> 모든 가중치를 동일하게 초기화하면 뉴런들이 동일하게 업데이트되어 학습이 진행되지 않음
> 랜덤 초기화로 각 뉴런이 서로 다른 가중치를 갖게 하여 독립적으로 학습 가능

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 단순한 구조 | 이해하기 쉽고 구현이 용이 — ANN의 기본 개념 학습에 필수 |
| 이진 분류 문제 해결 | 스팸 메일 분류, 이미지 이진 분류 등 두 클래스 중 하나로 분류하는 문제에 효과적 |
| 복잡한 신경망의 기초 | MLP, DNN 등 더 복잡한 신경망의 기본 단위 — 퍼셉트론의 원리를 층으로 쌓아 확장 |

### AND 연산 예제 코드

```python
import numpy as np

# AND 게이트 데이터
inputs  = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 0, 0, 1])

# 초기화
weights       = np.random.rand(2)   # 랜덤 초기화 — 대칭 깨기
bias          = np.random.rand(1)
learning_rate = 0.1
epochs        = 20

# 활성화 함수 (계단 함수)
def step_function(x):
    return 1 if x >= 0 else 0

# 학습
for epoch in range(epochs):
    for i in range(len(inputs)):
        total_input = np.dot(inputs[i], weights) + bias  # 가중합
        prediction  = step_function(total_input)          # 활성화 함수 적용
        error       = outputs[i] - prediction             # 오차 계산
        weights    += learning_rate * error * inputs[i]   # 가중치 업데이트
        bias       += learning_rate * error               # 편향 업데이트

# 검증
print("최종 가중치:", weights)
print("최종 편향:", bias)
for i in range(len(inputs)):
    total_input = np.dot(inputs[i], weights) + bias
    prediction  = step_function(total_input)
    print(f"입력: {inputs[i]}, 예측: {prediction}, 정답: {outputs[i]}")
```

| 코드 | 설명 |
| --- | --- |
| `np.dot(inputs[i], weights)` | 입력 벡터와 가중치 벡터의 점곱(dot product) — 각 요소를 곱하여 모두 더한 값 |
| `error = outputs[i] - prediction` | 실제값 - 예측값 → 양수면 예측 부족, 음수면 예측 과다, 0이면 정답 |
| `weights += learning_rate * error * inputs[i]` | 오차에 비례하여 가중치 조정 — 퍼셉트론 학습 알고리즘의 핵심 |

> 학습률(learning rate) 일반 기준
> - 소규모·단순 문제: 0.1 내외
> - 일반적인 딥러닝: 0.001 ~ 0.01 (Adam 옵티마이저 기본값은 0.001)
> - 너무 크면 발산, 너무 작으면 수렴이 느림
