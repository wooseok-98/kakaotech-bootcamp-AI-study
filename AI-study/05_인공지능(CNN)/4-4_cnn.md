# 4-4. CNN · Convolutional Layer · Pooling Layer · Flatten Layer

---

## CNN (Convolutional Neural Network)

**여러 개의 Convolutional Layer, Pooling Layer, Fully Connected Layer로 구성된 신경망**

- Convolutional: 합성곱 — 여러 요소를 합쳐 새로운 결과를 만드는 곱셈 연산
- 이미지의 공간적 구조를 유지하면서 특징을 자동으로 추출하고 학습

### 구조

| 단계 | 구성 요소 | 설명 |
| --- | --- | --- |
| Feature Learning | Convolutional Layer | 필터를 이용해 이미지의 특징 추출 |
| Feature Learning | Pooling Layer | 다운샘플링으로 연산량 감소 |
| Classification | Flatten Layer | 3D 특징 맵 → 1D 벡터 변환 |
| Classification | Fully Connected Layer | 최종 클래스 분류 |

| 주요 구성 요소 | 설명 |
| --- | --- |
| Convolutional Layer | 입력 데이터에 필터를 적용하고 활성화 함수를 반영하여 **특징을 추출**하는 레이어. 필터를 통해 이미지의 중요한 패턴을 감지 |
| Pooling Layer | Convolution Layer 다음에 위치하는 선택적 레이어. 특징 맵을 **다운샘플링**하여 연산량을 줄이고 모델의 일반화 성능을 높임. Max Pooling(최대값), Average Pooling(평균값) |
| Fully Connected Layer | 추출된 특징을 기반으로 최종 분류를 수행. 모든 뉴런이 서로 연결된 구조 |

### 기존 ANN(FCNN)의 문제점과 CNN의 해결

기존 Fully Connected Neural Network(FCNN)는 이미지를 1차원으로 Flatten하여 입력받기 때문에 두 가지 문제가 있음

| 문제점 | 설명 |
| --- | --- |
| 3차원 데이터 구조 손실 | 28×28 이미지를 `[1, 784]`로 변환할 때 이미지의 공간적 구조와 패턴 정보가 손실됨 |
| 학습 파라미터 폭증 | 은닉층 노드 50개×3층 기준 총 44,860개 파라미터 필요 — 계산 비용과 메모리 사용량 증가 |

**FCNN 파라미터 계산 예시** (28×28 입력, 은닉층 50×3개, 출력 10)

| 연결 구간 | 가중치 | 편향 | 합계 |
| --- | --- | --- | --- |
| 입력 → 은닉1 | 39,200 | 50 | 39,250 |
| 은닉1 → 은닉2 | 2,500 | 50 | 2,550 |
| 은닉2 → 은닉3 | 2,500 | 50 | 2,550 |
| 은닉3 → 출력 | 500 | 10 | 510 |
| **총합** | **44,700** | **160** | **44,860** |

CNN은 **가중치 공유(weight sharing)** 구조로 이를 해결 — 하나의 필터가 이미지 전체에 동일한 가중치를 적용하므로 파라미터 수를 크게 줄이고 과적합(overfitting)도 방지

### 사용 이유

| 이유 | 내용 |
| --- | --- |
| 공간적 구조 유지 | 이미지의 2D 공간 구조를 유지하며 특징을 효과적으로 추출 |
| 특징 추출 자동화 | 필터(커널)를 통해 자동으로 중요한 특징을 학습 |
| 파라미터 효율성 | 지역 연결과 가중치 공유로 파라미터 수 감소 |
| 우수한 성능 | 이미지 분류, 객체 탐지 등 시각적 데이터 처리에서 뛰어난 성능 |

### 코드 예시

#### TensorFlow (Keras)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

# 가상 데이터셋 생성 (32×32 RGB 이미지, 10 클래스)
num_classes = 10
x_train = np.random.random((1000, 32, 32, 3))
y_train = to_categorical(np.random.randint(num_classes, size=(1000, 1)), num_classes)
x_test = np.random.random((200, 32, 32, 3))
y_test = to_categorical(np.random.randint(num_classes, size=(200, 1)), num_classes)

# 모델 생성
model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax'),
])

# 컴파일 → 훈련 → 평가
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")
```

| 레이어 | 설명 |
| --- | --- |
| `Conv2D(32, (3,3), activation='relu')` | 32개 필터, 3×3 커널, ReLU 활성화로 특징 추출 |
| `MaxPooling2D((2,2))` | 2×2 영역에서 최대값 선택, 공간 크기 절반으로 감소 |
| `Flatten()` | 다차원 배열 → 1차원 변환 |
| `Dense(256, activation='relu')` | 256개 뉴런의 완전 연결층 |
| `Dense(num_classes, activation='softmax')` | 각 클래스 확률 출력 |

#### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 가상 데이터셋 생성
num_classes = 10
x_train = np.random.random((1000, 32, 32, 3)).astype(np.float32)
y_train = np.random.randint(num_classes, size=(1000,))
x_test = np.random.random((200, 32, 32, 3)).astype(np.float32)
y_test = np.random.randint(num_classes, size=(200,))

# permute(0, 3, 1, 2): (N, H, W, C) → (N, C, H, W) 형태로 변환
train_loader = DataLoader(
    TensorDataset(torch.tensor(x_train).permute(0, 3, 1, 2), torch.tensor(y_train)),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(x_test).permute(0, 3, 1, 2), torch.tensor(y_test)),
    batch_size=64, shuffle=False
)

# 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 훈련
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

# 평가
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        _, predicted = torch.max(model(inputs), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Accuracy: {correct / total * 100:.2f}%')
```

### 알아두면 좋은 정보

#### Global Average Pooling (GAP)

**특징 맵 전체 영역을 채널별로 평균내어 채널당 1개의 값만 남기는 기법**

예: 7×7×64 특징 맵 → GAP → 64차원 벡터

기존 FC층은 7×7×64(=3,136)를 500개 뉴런에 연결하면 약 156만 개의 가중치가 필요하지만, GAP는 추가 가중치 없이 채널별 평균만 계산

| 이점 | 설명 |
| --- | --- |
| 파라미터 감소 | 대형 FC층 대신 채널별 평균값(1×1×채널 수)만 남겨 모델 크기 축소 |
| 과적합 감소 | GAP 층 자체에 학습 파라미터가 없으므로 과적합 위험 감소 |
| 해석 용이 | 각 채널이 하나의 클래스 특징에 대응하도록 학습되어 설명 가능성 향상 |

> Max Pooling은 특정 위치의 최대값만 가져오지만, GAP(Global Average Pooling)는 이미지 전체 영역의 평균 활성도를 보므로 위치에 구애받지 않는 전역적 특성 요약이 가능

```python
# TensorFlow
import tensorflow as tf

x = tf.random.normal([1, 4, 4, 3])             # (배치, 높이, 너비, 채널)
y = tf.keras.layers.GlobalAveragePooling2D()(x)
print("GAP 결과 shape:", y.shape)               # (1, 3)

# PyTorch
import torch, torch.nn as nn

x = torch.rand(1, 3, 4, 4)                     # (배치, 채널, 높이, 너비)
y = nn.AdaptiveAvgPool2d((1, 1))(x).view(x.size(0), -1)
print("GAP 결과 shape:", y.shape)               # torch.Size([1, 3])
```

#### CNN vs. ANN vs. RNN

| 신경망 유형 | 정의 | 장점 | 단점 |
| --- | --- | --- | --- |
| CNN | 이미지/영상 처리에 특화. 공간적 구조를 유지하며 특징 추출 | 공간적 특징 학습, 계산 효율성 | 시계열 데이터 처리 부적합 |
| ANN | 가장 기본적인 신경망. 입력층·은닉층·출력층으로 구성 | 구현 간단, 다양한 문제에 적용 가능 | 공간적/시계열 구조 데이터 처리 비효율적 |
| RNN | 순차 데이터 처리에 특화. 이전 입력 정보를 기억하며 다음 입력에 반영 | 시간적 의존성 학습 가능 | 긴 시퀀스 처리 시 기울기 소실 문제 발생 |

---

## Convolutional Layer (합성곱 레이어)

**CNN에서 입력 데이터의 feature(특징)을 추출하는 레이어**

- Convolution Mask(필터/커널)를 사용해 합성곱 연산을 수행
- 합산 결과에 Bias를 더한 후 ReLU 활성화 함수를 적용
- Stride와 Padding 설정을 통해 출력 Feature Map의 크기를 조절

### 개념

| 용어 | 설명 |
| --- | --- |
| Convolution Mask (필터/커널) | 입력 데이터의 특정 패턴을 감지하는 작은 행렬. 각 셀에 학습 가능한 **가중치(weight)** 를 포함 |
| Convolution 연산 | 필터를 입력 데이터 위에서 이동하며 원소별 곱(element-wise multiplication) 후 합산. 주위의 지역 정보 특성까지 반영됨 |
| Convolved Feature (Feature Map) | Convolution 연산을 통해 얻어진 결과. 특징 맵이라고도 함 |
| Bias | 각 필터마다 학습되는 값. 합성곱 결과에 더해져 모델이 더 유연하게 학습할 수 있도록 도움 |
| Stride | 필터를 이동할 때의 간격. Stride=1이면 한 픽셀씩, 2 이상이면 더 띄엄띄엄 이동 |
| Padding | 입력 가장자리에 0을 추가하는 기법. Same Padding(입력 크기 유지) / Valid Padding(패딩 없음) |

**Convolution 연산 흐름**

1. Convolution Mask(필터)를 입력 데이터 위에서 Stride만큼 이동
2. 각 위치에서 필터와 입력 데이터의 해당 영역을 원소별 곱 후 합산
3. 합산 결과에 Bias를 더함
4. ReLU 활성화 함수 적용 (음수 → 0, 양수 → 그대로)
5. 연산 결과를 Feature Map에 저장

**입력/출력 구조 예시** (입력 7×7×3, 필터 2개)

| 구성 요소 | 크기 | 설명 |
| --- | --- | --- |
| Input Volume | 7×7×3 | 입력 이미지. ×3은 RGB 3채널 |
| Filter / Kernel | 3×3×3 | 패턴을 감지하는 필터. 입력 채널 수와 동일한 깊이 |
| Output Volume | 3×3×2 | 필터 2개 → 출력 채널 2개. 각 숫자는 가중합 결과 |

> 채널 수가 많아질수록 더 다양한 관점에서 이미지 분석 가능 — 32채널: 기본 패턴(선, 모서리), 64채널: 복잡한 질감, 128채널 이상: 물체 구성 요소 같은 고차원 특징

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 이미지의 중요한 부분을 찾아냄 | 필터를 사용해 경계선, 모서리 등 중요한 특징을 효과적으로 추출 |
| 연산량 감소 | 지역 연결된 필터를 사용하므로 FC Layer 대비 파라미터 수 크게 감소 |
| 단계적 특징 학습 | 초기 레이어는 단순한 특징(엣지), 후속 레이어는 복잡한 특징(모양, 객체)을 점진적으로 학습 |

### 코드 예시 (CIFAR-10 유사 데이터)

#### TensorFlow (Keras)

```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

# 가상 데이터 생성 (32×32 RGB, 10 클래스)
num_samples, num_classes = 10000, 10
X = np.random.rand(num_samples, 32, 32, 3).astype(np.float32)
y = np.random.randint(num_classes, size=num_samples)

train_size = int(0.8 * num_samples)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(train_size).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

# 모델 정의
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes),
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Validation Accuracy: {val_accuracy:.2f}")
```

| 파라미터 | 설명 |
| --- | --- |
| `filters` (out_channels) | 학습할 필터 개수. 필터 1개가 특징 1개를 학습 |
| `kernel_size` | 커널(Convolution Mask) 크기 |
| `padding='same'` | 출력 크기가 입력과 동일하게 유지되도록 0을 덧댐 |
| `from_logits=True` | Dense 출력이 소프트맥스를 거치지 않은 로짓일 때 설정 |

#### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# 가상 데이터 생성
num_samples, num_classes = 10000, 10
X = np.random.rand(num_samples, 3, 32, 32).astype(np.float32)  # (N, C, H, W)
y = np.random.randint(num_classes, size=num_samples).astype(np.int64)

dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # (N, 32, 16, 16)
        x = self.pool(torch.relu(self.conv2(x)))  # (N, 64, 8, 8)
        x = x.view(-1, 64 * 8 * 8)               # Flatten → (N, 4096)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 훈련
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

# 평가
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        _, predicted = torch.max(model(inputs), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Validation Accuracy: {100 * correct / total:.2f}%")
```

| 파라미터 | 설명 |
| --- | --- |
| `in_channels` | 입력 채널 수 (흑백=1, RGB=3, 이전 레이어의 out_channels와 일치) |
| `out_channels` | 학습할 필터 개수 |
| `padding=1` | 입력 가장자리에 0을 1픽셀씩 추가 → 출력 크기가 입력과 동일하게 유지 |
| `x.view(-1, 64*8*8)` | (N, 64, 8, 8) → (N, 4096) 변환. `-1`은 배치 크기 자동 계산 |

> `torch.max(outputs, 1)`은 `(최대값, 최대값의 인덱스)` 두 값을 반환. `_, predicted = torch.max(outputs, 1)`에서 `_`는 최대값(사용 안 함), `predicted`는 예측된 클래스 인덱스

### 알아두면 좋은 정보

#### Convolution Mask(필터/커널) 크기와 개수

| 크기 | 특징 |
| --- | --- |
| 3×3 | 가장 일반적. 미세한 특징 추출, 적은 파라미터 |
| 5×5 | 더 넓은 영역의 특징 포착 |
| 7×7 | 첫 레이어에서 넓은 수용 영역 확보 시 사용 |

- 필터 개수가 많을수록 더 많은 특징을 추출할 수 있지만 연산량도 증가
- 일반적으로 레이어가 깊어질수록 필터 개수를 늘림 (32 → 64 → 128)

---

## Pooling Layer (풀링 레이어)

**CNN에서 입력 특징 맵의 공간 크기를 줄여 계산량을 감소시키고, 중요한 특징을 추출하며, 과적합을 방지하는 역할을 하는 레이어**

### 개념

풀링은 특성 맵(feature map)의 크기를 줄이고 중요한 정보를 추려내는 연산으로, 대표적으로 맥스 풀링과 평균 풀링이 사용됨

#### 맥스 풀링 (Max Pooling)

풀링 영역(예: 2×2) 내에서 **가장 큰 값**을 대표값으로 선택하는 방식

- 에지, 경계 등 뚜렷한 패턴을 강조하는 데 유리
- 작은 노이즈나 위치 변동의 영향을 줄임
- 예: `[1, 5, 3, 2]` → **5** 선택

#### 평균 풀링 (Average Pooling)

풀링 영역 내의 모든 값의 **평균**을 대표값으로 계산하는 방식

- 구역 전체의 통계적 특성을 고르게 반영
- 극단적인 값을 완화하여 전체 분포를 안정적으로 유지
- 예: `[1, 5, 3, 2]` → `(1+5+3+2)/4 = 2.75`

> 에지(Edge): 이미지에서 픽셀 값이 급격하게 변하는 경계. 물체의 윤곽선이나 색상·밝기 변화가 두드러지는 부분으로, 특징 추출에 중요한 단서를 제공

**맥스 풀링 vs 평균 풀링 비교**

| 구분 | 맥스 풀링 (Max Pooling) | 평균 풀링 (Average Pooling) |
| --- | --- | --- |
| 정의 | 풀링 영역 내에서 가장 큰 값을 선택 | 풀링 영역 내의 모든 값의 평균을 계산 |
| 강조되는 특징 | 에지, 경계 등 두드러지는 값 | 전체 분포와 통계적 특성 |
| 노이즈 처리 | 최댓값만 뽑으므로 노이즈 영향 상대적으로 작음 | 평균화로 노이즈와 이상치를 상쇄 |
| 활용 분야 | 이미지 분류, 물체 인식 | 국소 영역의 전반적 정보 유지, GAP를 통한 최종 분류 |
| 장점 | 뚜렷한 특징 강조, 학습 빠름, 위치 불변성 | 전체 데이터 균등 반영, 노이즈 완화, 안정성 기여 |
| 단점 | 구역 내부 분포 정보 소실, 극단값에만 집중 | 뚜렷한 특징 부각 효과가 낮을 수 있음 |

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 계산량 감소 | 입력 특징 맵의 크기를 줄여 연산량과 메모리 사용량 감소 |
| 불변성 확보 | 위치 변동, 크기 변화, 약간의 회전에 대해 특징을 유지 — 맥스 풀링으로 국소적으로 가장 중요한 특징만 남기면 작은 위치 이동에 덜 민감해짐 |
| 과적합 방지 | 데이터 차원을 줄이고 미세한 변동을 뭉뚱그려 표현하여 모델 복잡도 낮춤 |
| 중요한 특징 추출 | 노이즈나 세부 변동을 제거하고 핵심 패턴을 부각하여 분류/인식 작업에 유리 |

### 코드 예시

#### TensorFlow (Keras)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

num_classes = 10
x_train = np.random.random((1000, 32, 32, 3))
y_train = to_categorical(np.random.randint(num_classes, size=(1000, 1)), num_classes)
x_test = np.random.random((200, 32, 32, 3))
y_test = to_categorical(np.random.randint(num_classes, size=(200, 1)), num_classes)

model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),          # 32×32 → 16×16
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),          # 16×16 → 8×8
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),          # 8×8 → 4×4
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")
```

#### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

num_classes = 10
x_train = np.random.random((1000, 32, 32, 3)).astype(np.float32)
y_train = np.random.randint(num_classes, size=(1000,))
x_test = np.random.random((200, 32, 32, 3)).astype(np.float32)
y_test = np.random.randint(num_classes, size=(200,))

train_loader = DataLoader(
    TensorDataset(torch.tensor(x_train).permute(0, 3, 1, 2), torch.tensor(y_train)),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(x_test).permute(0, 3, 1, 2), torch.tensor(y_test)),
    batch_size=64, shuffle=False
)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 공간 크기 절반 감소
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # (N, 32, 16, 16)
        x = self.pool(torch.relu(self.conv2(x)))  # (N, 64, 8, 8)
        x = self.pool(torch.relu(self.conv3(x)))  # (N, 128, 4, 4)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        _, predicted = torch.max(model(inputs), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Accuracy: {correct / total * 100:.2f}%')
```

---

## Flatten Layer (플래튼 레이어)

**다차원 배열 형태의 입력 데이터를 1차원 배열로 변환하여 Fully Connected Layer(Dense Layer)에 입력할 수 있도록 하는 신경망 레이어**

### 개념

CNN은 Conv2D, MaxPooling2D 레이어를 거치며 4D 텐서 `(Batch, Height, Width, Channels)`를 출력함. Fully Connected(Dense) Layer는 `(Batch, Features)` 형태의 2D 입력만 받으므로, 두 레이어 사이에 Flatten이 필요함

**shape 변환 예시** (28×28 흑백 이미지, 배치 크기 32)

| 단계 | shape | 설명 |
| --- | --- | --- |
| 입력 이미지 | `(32, 28, 28, 1)` | 배치 32개, 28×28, 1채널(Grayscale) |
| Conv2D + Pooling 후 | `(32, 14, 14, 8)` | 8개 필터 적용, 풀링으로 공간 크기 절반 감소 |
| Flatten 후 | `(32, 1568)` | 14×14×8 = 1568개의 1D 특징 벡터로 변환 |

- Batch 차원(32)은 그대로 유지
- 나머지 차원(Height, Width, Channels)을 모두 곱해 1D 벡터로 변환
- 배치 전체가 아니라 **각 샘플(이미지) 하나하나**가 1D 벡터가 되는 것

> FCNN의 Dense Layer는 개별 샘플 단위로 연산하지만 배치 단위로 병렬 처리함. `(Batch, Features)` 형태의 입력은 각 샘플이 1D 벡터인 상태이므로 그대로 Dense Layer에 입력 가능

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 다차원 데이터를 1차원으로 변환 | 이미지나 시계열 데이터에서 추출된 다차원 특징을 Dense Layer가 처리할 수 있는 형태로 변환 |
| CNN과 FCNN의 연결 역할 | 합성곱·풀링으로 추출한 특징 맵을 최종 분류 레이어에 연결하는 필수 인터페이스 |
| 학습 파라미터 없음 | 단순 형태 변환만 수행하므로 별도의 가중치·편향 학습 없이 구조를 단순하게 유지 |
| 코드 편의성 | 별도의 데이터 변환 과정 없이 자동으로 형태를 조정 |

### 코드 예시

#### TensorFlow (Keras)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

num_classes = 10
x_train = np.random.random((1000, 32, 32, 3))
y_train = to_categorical(np.random.randint(num_classes, size=(1000, 1)), num_classes)
x_test = np.random.random((200, 32, 32, 3))
y_test = to_categorical(np.random.randint(num_classes, size=(200, 1)), num_classes)

model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),                     # (N, 6, 6, 64) → (N, 2304)
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")
```

#### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

num_classes = 10
x_train = np.random.random((1000, 32, 32, 3)).astype(np.float32)
y_train = np.random.randint(num_classes, size=(1000,))
x_test = np.random.random((200, 32, 32, 3)).astype(np.float32)
y_test = np.random.randint(num_classes, size=(200,))

train_loader = DataLoader(
    TensorDataset(torch.tensor(x_train).permute(0, 3, 1, 2), torch.tensor(y_train)),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(x_test).permute(0, 3, 1, 2), torch.tensor(y_test)),
    batch_size=64, shuffle=False
)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()        # (N, 128, 4, 4) → (N, 2048)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        _, predicted = torch.max(model(inputs), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Accuracy: {correct / total * 100:.2f}%')
```

---

## Model Architecture (모델 아키텍처)

**신경망에서 각 레이어의 구성, 연결 방식, 활성화 함수, 입력 및 출력 형태 등 전체 모델의 구조**

| 용어 | 설명 |
| --- | --- |
| Model | 여러 개의 층(layer)과 뉴런(neuron), 이들 사이의 연결을 나타내는 가중치(weights)와 편향(biases)으로 구성된 집합체 — 즉, 신경망 |
| Architecture | 소프트웨어 시스템의 구조 |
| Model Architecture | 신경망에서 각 레이어의 구성, 연결 방식, 활성화 함수, 입력 및 출력 형태 등 전체 모델의 구조 |

딥러닝의 목적: **복잡한 데이터에서 높은 수준의 특징을 자동으로 학습해 weight를 조정하는 것**

- Feed-Forward(순전파) 과정을 통해 나온 결과를 정답과 비교하고, Backpropagation(역전파) 과정을 거쳐 weight를 조정하는 과정이 자동으로 이루어짐
- 모델 파일에는 학습된 **가중치와 편향만 저장** — 모델 구조는 코드로 정의되어 있으므로 별도 저장 불필요

> **복잡한 데이터**: 다차원적이며 다양한 패턴과 변형을 포함하는 데이터 (이미지, 음성, 자연어 텍스트, 시계열 등)
>
> **높은 수준의 특징(high-level features)**: 원시 데이터의 저수준 특징을 기반으로 추출된 추상적이고 의미 있는 패턴
> - 이미지: 저수준(엣지, 텍스처) → 중간(원, 사각형) → 고수준(사람 얼굴, 물체)
> - 텍스트: 저수준(단어) → 중간(문장 구조) → 고수준(문맥적 의미, 주제)

### 에포크별 학습 과정

Training Data 3개, Validation Data 있음, 5 Epoch 가정 시

**1 Epoch 내 과정**

1. Training Data마다 각각 다른 비선형 그래프가 그려지고, 학습률(Learning Rate)을 조정하며 최적의 Loss 값을 탐색
2. 모든 Training Data 학습 완료 후 Validation Data로 성능 검증
   - 모델이 Validation Data에 대한 예측을 수행하고 실제 레이블과 비교
   - Loss 값과 정확도(accuracy) 계산
3. 해당 Epoch의 Loss와 정확도 출력, 학습된 모델 저장

> 학습 출력의 `15/15`에서 첫 번째 숫자는 현재 배치 번호, 두 번째 숫자는 총 배치 수. 즉 해당 에포크의 마지막 배치(64개씩 나누어 총 15번의 배치)를 처리 중임을 의미

**전체 5 Epoch 과정**

1. Epoch를 한 번 돌 때마다 모델이 하나씩 생성
2. 모든 Epoch 완료 후 Test Data로 평가 — Epoch 수만큼 반복
3. 가장 성능이 좋은 모델이 최종 모델로 선택

**손실(Loss)과 정확도(Accuracy) 평가 기준**

| 지표 | 기준 |
| --- | --- |
| 손실(Loss) | 0에 가까울수록 좋음. 일반적으로 0.1 이하이면 좋은 성능. 단, 과적합 시 손실이 낮아도 테스트 성능이 낮을 수 있음 |
| 정확도(Accuracy) | 높을수록 좋음. 이진 분류: 90% 이상, 다중 클래스 분류: 80% 이상이면 좋은 성능으로 간주 |

> 평가 지표(metrics)는 `accuracy` 외에도 Precision(정밀도), Recall(재현율) 등을 사용할 수 있음

### 사용 이유

| 이유 | 내용 |
| --- | --- |
| 성능 최적화 | 적절한 모델 아키텍처를 선택하면 학습 능력과 예측 성능을 극대화 |
| 효율성 향상 | 복잡한 계산을 효율적으로 처리할 수 있는 구조 설계로 훈련 시간과 자원 소모 감소 |
| 일반화 능력 | 다양한 데이터셋에 대해 좋은 성능을 보이는 모델 설계 가능 |

### 코드 예시

#### TensorFlow (Keras)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

# 가상 데이터셋 생성
num_classes = 10
x_train = np.random.random((1000, 32, 32, 3))
y_train = to_categorical(np.random.randint(num_classes, size=(1000, 1)), num_classes)
x_test = np.random.random((200, 32, 32, 3))
y_test = to_categorical(np.random.randint(num_classes, size=(200, 1)), num_classes)

# 모델 아키텍처 정의
model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu'),      # Convolutional Layer
    MaxPooling2D(pool_size=(2, 2)),              # Pooling Layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),                                   # Flatten Layer
    Dense(256, activation='relu'),               # Fully Connected Layer
    Dense(num_classes, activation='softmax'),    # Output Layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")
```

#### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 가상 데이터셋 생성
num_classes = 10
x_train = np.random.random((1000, 32, 32, 3)).astype(np.float32)
y_train = np.random.randint(num_classes, size=(1000,))
x_test = np.random.random((200, 32, 32, 3)).astype(np.float32)
y_test = np.random.randint(num_classes, size=(200,))

train_loader = DataLoader(
    TensorDataset(torch.tensor(x_train).permute(0, 3, 1, 2), torch.tensor(y_train)),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(x_test).permute(0, 3, 1, 2), torch.tensor(y_test)),
    batch_size=64, shuffle=False
)

# 모델 아키텍처 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # Convolutional Layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)          # Pooling Layer
        self.flatten = nn.Flatten()                                 # Flatten Layer
        self.fc1 = nn.Linear(128 * 4 * 4, 256)                    # Fully Connected Layer
        self.fc2 = nn.Linear(256, num_classes)                     # Output Layer

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        _, predicted = torch.max(model(inputs), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Accuracy: {correct / total * 100:.2f}%')
```

### 알아두면 좋은 정보

#### 딥러닝 구조 전반

| 단계 | 설명 |
| --- | --- |
| 1. Feed-Forward (순전파) | 입력 데이터가 가중치를 거쳐 선형 변환된 후, 활성화 함수를 통해 비선형성을 부여받고 다음 레이어로 전달. 최종 출력층에서 예측 결과 생성 |
| 2. Loss Function (손실 함수) | 실제 데이터와 예측 데이터 간의 오차(Loss)를 계산 |
| 3. Backpropagation (역전파) | 오차(Loss) 값을 미분하여 기울기(Gradient)를 계산. 가중치(Weight)를 조정하면 기울기 값도 변하며, 오차를 줄이는 방향으로 점진적으로 업데이트 |
| 4. Optimizing (조정) | 역전파를 통해 계산한 가중치의 기울기를 활용해 실제 가중치를 조정 |
| 5. 반복 | 위 과정을 정해진 에포크(epoch)만큼 반복 |
