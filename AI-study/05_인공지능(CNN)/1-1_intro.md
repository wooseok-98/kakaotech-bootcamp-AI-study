# 1-1. TensorFlow · Keras · PyTorch · Kaggle

---

## TensorFlow

**Google Brain 팀에서 개발한 오픈소스 머신러닝 및 딥러닝 라이브러리**

모든 데이터를 텐서(Tensor)라는 다차원 배열로 다루며, 수학적 연산을 효율적으로 처리하고 자동 미분, GPU·TPU 가속을 지원하는 연산 엔진 역할을 함

| 특징 | 설명 |
| --- | --- |
| 성능 | 컴퓨팅 성능이 뛰어나고 대규모 데이터 처리 및 복잡한 연산을 빠르게 수행 가능 |
| 확장성 | 다양한 플랫폼과 언어를 지원하여 높은 확장성과 유연성 제공 |
| 시각화 | TensorBoard를 통해 모델의 학습 과정과 성능을 시각화하고 디버깅 가능 |
| 커뮤니티 | 활발한 커뮤니티 지원과 다양한 튜토리얼, 예제 코드 제공 |

### 동작 방식

| 순서 | 동작 | 설명 |
| --- | --- | --- |
| 1 | 텐서 표현 | 모든 데이터는 다차원 배열(텐서) 형태로 관리되며, 연산의 입력과 출력으로 사용 |
| 2 | 연산과 그래프 | 연산을 노드, 데이터 흐름을 엣지로 표현하여 계산을 그래프로 모델링. TF 2.x는 즉시 실행(Eager Execution)을 기본으로 하며, 필요 시 `tf.function`으로 그래프 최적화 수행 가능 |
| 3 | 자동 미분 | `tf.GradientTape`를 통해 기울기(gradient)를 자동으로 계산하여 파라미터를 효율적으로 갱신 |
| 4 | 모델 구성 | `tf.keras` 등 고수준 API를 이용해 모델을 정의하고, 학습·평가·예측 과정을 손쉽게 구현 |
| 5 | 학습과 최적화 | 손실 함수와 옵티마이저(SGD, Adam 등)로 파라미터를 반복 업데이트하며, GPU/TPU 가속 활용 |
| 6 | 추론과 배포 | TensorFlow Serving·Lite·JS 등을 통해 모바일, 서버, 웹 등 다양한 환경에 배포 가능 |

### TensorFlow 1.x vs 2.x

| 변경점 | 1.x | 2.x |
| --- | --- | --- |
| 즉시 실행 | 연산 실행 전 그래프(Session)를 정의해야 함 | 코드를 작성하면 바로 실행 — 디버깅이 쉬워짐 |
| Keras 통합 | Keras가 외부 라이브러리였으며, 별도 설치 필요 | `tf.keras`가 공식 API로 통합 |
| 그래프 실행 | 모든 연산이 정적인 그래프에서 실행 | 즉시 실행 기본, `@tf.function`으로 그래프 최적화 가능 |

```python
# TensorFlow 1.x — 정적 그래프, Session 필요
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
with tf.Session() as sess:
    result = sess.run(z, feed_dict={x: 3, y: 4})  # 7.0

# TensorFlow 2.x — 즉시 실행, Session 불필요
x = tf.constant(3.0)
y = tf.constant(4.0)
z = x + y
print(z)  # tf.Tensor(7.0, shape=(), dtype=float32)
```

---

## Keras

**TensorFlow 위에서 동작하는 고수준 딥러닝 API**

층(layer)을 쌓는 방식으로 모델을 설계할 수 있고, 코드 몇 줄로 학습·평가·예측을 처리할 수 있을 정도로 사용자 친화적

현재는 TensorFlow에 통합되어 `tf.keras` 형태로 제공됨

| 특징 | 설명 |
| --- | --- |
| 고수준 API | 직관적이고 간편한 API로 딥러닝 모델을 쉽게 설계·구축·훈련 가능 |
| 빠른 프로토타이핑 | 빠르게 모델을 실험할 수 있어 연구와 개발에 유용 |
| 사용자 친화적 | 초보자도 쉽게 딥러닝 모델을 만들고 사용할 수 있는 인터페이스 제공 |

### 동작 방식

| 순서 | 동작 | 설명 |
| --- | --- | --- |
| 1 | 모델 정의 | 층을 순차적으로 쌓거나 다양한 연결 구조로 신경망 모델 구성 |
| 2 | 모델 컴파일 | 손실 함수, 최적화 방법, 평가 기준을 지정하여 학습 준비 |
| 3 | 모델 훈련 | 데이터를 반복 학습하며 자동 미분과 최적화로 가중치 조정 |
| 4 | 모델 평가 | 새로운 데이터로 모델 성능 측정 |
| 5 | 모델 예측 | 학습된 모델로 입력 데이터에 대한 출력 생성 |
| 6 | 학습 관리 | 조기 종료, 모델 저장, 모니터링 등 학습 과정을 효율적으로 제어 |
| 7 | 배포와 확장 | 다양한 환경에 배포하거나 사용자 정의 기능으로 확장 가능 |

> TensorFlow는 딥러닝의 핵심 연산을 수행하는 **엔진**, Keras는 TensorFlow를 쉽게 사용할 수 있도록 돕는 **인터페이스**

### TensorFlow + Keras 사용 예시

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 가상 데이터셋 생성
np.random.seed(0)
tf.random.set_seed(0)
train_data   = np.random.rand(1000, 10)
train_labels = np.random.randint(2, size=(1000, 1))
test_data    = np.random.rand(100, 10)
test_labels  = np.random.randint(2, size=(100, 1))

# 모델 정의
model = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=(10,)),  # 은닉층
    layers.Dense(1,  activation='sigmoid')                    # 출력층 (이진 분류)
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
model.fit(train_data, train_labels, epochs=10, batch_size=32)

# 모델 평가
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {test_acc}")

# 예측
predictions = model.predict(np.asarray(test_data, dtype=np.float32))
print(f"첫 번째 샘플 예측값: {predictions[0]}")
```

---

## PyTorch

**딥러닝 모델을 쉽게 구축하고 학습할 수 있도록 설계된 파이썬 기반의 오픈소스 딥러닝 라이브러리**

텐서(Tensor) 데이터 구조와 동적 계산 그래프를 사용하며, Autograd 모듈로 역전파를 자동으로 수행

| 특징 | 설명 |
| --- | --- |
| 직관적인 코딩 | 파이썬과 유사한 문법으로 간결한 코딩 가능 |
| 동적 계산 그래프 | 실행 시점에 그래프를 생성 — 실시간 수정 및 디버깅 용이 |
| 자동 미분 | Autograd 모듈이 역전파를 위한 기울기를 자동으로 계산 |
| 풍부한 생태계 | 연구 및 실무에서 널리 사용되며, 최신 딥러닝 기술 적용에 적합 |

### 동작 방식

| 순서 | 동작 | 설명 |
| --- | --- | --- |
| 1 | 텐서 정의 및 연산 | PyTorch 텐서로 데이터와 가중치를 저장하고 연산 수행 |
| 2 | 동적 계산 그래프 생성 | 연산마다 새로운 계산 그래프를 동적으로 생성하여 역전파 시 기울기 자동 계산 |
| 3 | 자동 미분 | Autograd로 역전파를 통해 기울기를 자동으로 계산하고 파라미터 업데이트 |
| 4 | 모델 정의 및 학습 | `torch.nn` 모듈로 신경망을 정의하고 옵티마이저로 모델 학습 |

### PyTorch 사용 예시

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 가상 데이터셋 생성
np.random.seed(0)
train_data   = np.random.rand(1000, 10).astype(np.float32)
train_labels = np.random.randint(2, size=(1000, 1)).astype(np.float32)
test_data    = np.random.rand(100, 10).astype(np.float32)
test_labels  = np.random.randint(2, size=(100, 1)).astype(np.float32)

# DataLoader 구성
train_loader = DataLoader(TensorDataset(torch.tensor(train_data), torch.tensor(train_labels)), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(torch.tensor(test_data),  torch.tensor(test_labels)),  batch_size=32, shuffle=False)

# 모델 정의
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = SimpleNN()

# 손실 함수 & 옵티마이저
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 훈련
for epoch in range(10):
    model.train()
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_data).squeeze()
        loss = criterion(outputs, batch_labels.squeeze())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 모델 평가
model.eval()
correct = total = 0
with torch.no_grad():
    for batch_data, batch_labels in test_loader:
        predicted = (model(batch_data).squeeze() > 0.5).float()
        total   += batch_labels.size(0)
        correct += (predicted == batch_labels.squeeze()).sum().item()
print(f"Test accuracy: {correct / total:.4f}")
```

---

## TensorFlow vs PyTorch

| 항목 | TensorFlow | PyTorch |
| --- | --- | --- |
| 계산 그래프 | 정적 (2.x부터 즉시 실행 지원) | 동적 (실행 시점에 생성) |
| 디버깅 | 상대적으로 복잡 | 직관적 |
| 유연성 | 낮음 | 높음 |
| 학습 곡선 | 가파름 | 완만함 |
| 배포 | 강력한 지원 (TF Serving, TF Lite 등) | 상대적으로 미흡 |
| 주요 사용처 | 상용 애플리케이션, 대규모 배포 | 연구·실험 환경 |

---

## Kaggle

**전 세계 데이터 연구자들이 데이터를 분석할 수 있도록 대회를 개최하고, 분석 내용을 토론할 수 있는 커뮤니티를 제공하는 플랫폼**

다양한 데이터셋과 코드 노트북(Kernel)을 공유하여 학습 자료와 실습 환경 제공

| 이유 | 설명 |
| --- | --- |
| 실제 데이터 접근 | 다양한 도메인의 실제 데이터를 분석하며 실무 경험 축적 가능 |
| 커뮤니티 및 협업 | 다양한 수준의 데이터 과학자들이 아이디어를 공유하고 토론 |
| 코드 공유 | 다른 사람의 코드 노트북을 보고 학습하여 문제 해결 능력 향상 |

### Colab에서 Kaggle 데이터셋 사용하기

**1단계 — Kaggle API Key 발급**

Kaggle 사이트 → Settings → API → `Create New Token` → `kaggle.json` 파일 다운로드

**2단계 — Colab에 API Key 업로드**

```python
from google.colab import files
files.upload()  # kaggle.json 업로드

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

| 코드 | 설명 |
| --- | --- |
| `files.upload()` | Colab에 로컬 파일 업로드 |
| `mkdir -p ~/.kaggle` | Kaggle 설정 디렉토리 생성 (`-p`: 상위 폴더 없어도 생성, 이미 존재해도 오류 없음) |
| `cp kaggle.json ~/.kaggle/` | API Key 파일을 설정 디렉토리로 복사 |
| `chmod 600` | 파일 접근 권한을 소유자 읽기·쓰기 전용으로 설정 |

**3단계 — 데이터셋 다운로드 및 압축 해제**

```python
# 데이터셋 다운로드 (kaggle.com/datasets/ 뒷부분을 -d 옵션에 입력)
!kaggle datasets download -d sumn2u/riped-and-unriped-tomato-dataset

# 압축 해제
!unzip riped-and-unriped-tomato-dataset.zip -d riped-and-unriped-tomato
```

**4단계 — 파일 목록 확인**

```python
import os

data_dir = 'riped-and-unriped-tomato'
for root, dirs, files in os.walk(data_dir):
    for file in files:
        print(os.path.join(root, file))
```
