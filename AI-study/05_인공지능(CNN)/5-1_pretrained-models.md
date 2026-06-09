# 5-1. Pre-trained Model · ResNet · VGG16

---

## 사전 훈련 모델 (Pre-trained Model)

**대규모 데이터셋으로 미리 학습이 완료된 모델**

- ImageNet(1,400만 장, 1,000 클래스)과 같은 대규모 데이터셋으로 사전 학습
- 저수준 특징(선, 엣지)부터 고수준 특징(물체, 얼굴)까지 이미 학습된 가중치 보유
- 모델 파일에는 학습된 **가중치(weight)와 편향(bias)만 저장** — 모델 구조는 코드로 정의

### 특징

| 항목 | 설명 |
| --- | --- |
| 시간 절약 | 처음부터 학습하지 않아도 되므로 훈련 시간과 계산 자원 절약 |
| 성능 향상 | 대규모 데이터로 학습한 일반적인 특징을 재사용하여 성능 향상 |
| 데이터 부족 극복 | 소규모 데이터셋에서도 높은 성능 달성 가능 |
| 자원 효율성 | 컴퓨팅 자원을 절약하여 더 효율적으로 모델 활용 |

### 대표 모델

| 모델 | 특징 |
| --- | --- |
| VGG16 / VGG19 | 단순하고 균일한 3×3 구조. 교육·특징 추출 목적에 많이 사용 |
| ResNet (18/34/50/101/152) | Skip Connection으로 깊은 네트워크 학습 가능 |
| InceptionNet | 병렬 합성곱 경로를 가진 Inception Module 사용 |
| MobileNet | 경량화 설계. 모바일·엣지 환경에 최적화 |
| EfficientNet | 깊이·너비·해상도를 균형 있게 스케일링 |

> 모델의 성능을 평가할 때는 **정확도(accuracy)와 손실(loss)을 함께** 비교해야 함 — 정확도만 높아도 손실 값이 크다면 실제 성능이 좋다고 볼 수 없음

### 코드 예시

#### TensorFlow (Keras)

```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np

num_classes = 10

x_train = np.random.random((100, 224, 224, 3)).astype(np.float32)
y_train = to_categorical(np.random.randint(num_classes, size=(100, 1)), num_classes)
x_test = np.random.random((20, 224, 224, 3)).astype(np.float32)
y_test = to_categorical(np.random.randint(num_classes, size=(20, 1)), num_classes)

# 사전 훈련된 EfficientNetB0 로드 (분류 헤드 제외)
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax'),
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=16, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")
```

| 코드 | 설명 |
| --- | --- |
| `weights='imagenet'` | ImageNet으로 사전 훈련된 가중치 로드 |
| `include_top=False` | 최상단 분류 레이어 제외 — 커스텀 분류 헤드 추가용 |
| `Adam(learning_rate=0.0001)` | 사전 훈련 모델 활용 시 낮은 학습률 권장 |

#### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import numpy as np

num_classes = 10

x_train = np.random.random((100, 3, 224, 224)).astype(np.float32)
y_train = np.random.randint(num_classes, size=(100,))
x_test = np.random.random((20, 3, 224, 224)).astype(np.float32)
y_test = np.random.randint(num_classes, size=(20,))

loader = DataLoader(
    TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=16
)

# 사전 훈련된 EfficientNet-B0 로드
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# 분류 헤드 교체
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model.train()
for epoch in range(3):
    running_loss = 0.0
    for inputs, labels in loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(loader):.4f}')
```

| 코드 | 설명 |
| --- | --- |
| `EfficientNet_B0_Weights.IMAGENET1K_V1` | ImageNet 사전 훈련 가중치 로드 |
| `model.classifier[1] = nn.Linear(...)` | 마지막 분류층을 새 클래스 수에 맞게 교체 |

---

## ResNet (Residual Network)

**Skip Connection(잔차 연결)을 사용하여 깊은 신경망에서도 기울기 소실 문제를 해결하는 모델**

- 2015 ILSVRC(ImageNet) 우승 모델
- 층이 깊어질수록 오히려 성능이 떨어지는 **Degradation Problem**을 해결

### Skip Connection (잔차 연결)

```
일반 네트워크: x → [Conv → BN → ReLU → Conv → BN] → F(x)
ResNet:         x → [Conv → BN → ReLU → Conv → BN] → F(x) + x → ReLU
```

입력 x를 변환하지 않고 출력에 직접 더함 → 역전파 시 기울기가 Skip Connection 경로를 통해 직접 전달되므로 깊은 층에서도 기울기가 0에 수렴하지 않음

| 개념 | 설명 |
| --- | --- |
| Residual (잔차) | 입력과 출력의 차이 — F(x) = H(x) - x |
| Skip Connection | 입력을 변환 없이 출력에 직접 더하는 연결 (Residual connection이라고도 함) |
| Identity Mapping | 입력을 그대로 전달 — 기울기 소실 문제 완화 |
| Bottleneck Block | 1×1 → 3×3 → 1×1 Conv 구조. 50층 이상에서 사용하여 파라미터 절약 |

### ResNet 변형

| 모델 | 층 수 | 특징 |
| --- | --- | --- |
| ResNet-18 | 18층 | 기본형. 경량화 환경에 적합 |
| ResNet-34 | 34층 | Basic Block 사용 |
| ResNet-50 | 50층 | Bottleneck Block 사용. 실무에서 가장 많이 사용 |
| ResNet-101/152 | 101/152층 | 더 높은 정확도. 대형 데이터셋에 적합 |

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 기울기 소실 문제 해결 | 잔차 연결로 깊은 네트워크에서도 안정적인 학습 |
| 높은 성능 | 이미지 인식·객체 탐지·분류 작업에서 검증된 성능 |
| 사전 훈련 가중치 활용 | 전이 학습의 기반 모델로 널리 사용 |
| 확장성 | ResNet-18~152 등 다양한 깊이로 상황에 맞게 선택 가능 |

### 코드 예시 (미니퀘스트)

#### TensorFlow (Keras)

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np

num_classes = 10

x_train = np.random.random((100, 224, 224, 3)).astype(np.float32)
y_train = np.random.randint(num_classes, size=(100,))
x_test = np.random.random((20, 224, 224, 3)).astype(np.float32)
y_test = np.random.randint(num_classes, size=(20,))

# 사전 훈련된 ResNet50 로드 (분류 헤드 제외)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # 사전 훈련 가중치 고정

# 분류 헤드 추가 (함수형 API)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=16, validation_split=0.1)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")
```

| 코드 | 설명 |
| --- | --- |
| `weights='imagenet'` | ImageNet으로 사전 훈련된 가중치 로드 |
| `include_top=False` | 최상단 분류 레이어 제외 — 커스텀 분류 헤드 추가용 |
| `base_model.trainable = False` | 사전 훈련 가중치 고정 |
| `GlobalAveragePooling2D()` | 특징 맵을 채널별 평균으로 압축 — FC층 파라미터 수 절감 |

#### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

num_classes = 10
x_train = torch.rand(100, 3, 224, 224)
y_train = torch.randint(num_classes, (100,))
x_test = torch.rand(20, 3, 224, 224)
y_test = torch.randint(num_classes, (20,))

loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16)

# 사전 훈련된 ResNet50 로드
model = models.resnet50(pretrained=True)

# 분류 헤드 교체 (1000 클래스 → num_classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# FC층 외 가중치 고정
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

model.train()
for epoch in range(3):
    running_loss = 0.0
    for inputs, labels in loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(loader):.4f}')

model.eval()
with torch.no_grad():
    _, predicted = torch.max(model(x_test), 1)
    accuracy = (predicted == y_test).float().mean().item()
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

| 코드 | 설명 |
| --- | --- |
| `pretrained=True` | 사전 훈련된 가중치 로드 |
| `model.fc = nn.Linear(...)` | 마지막 완전 연결층을 새 클래스 수에 맞게 교체 |
| `param.requires_grad = False` | 해당 파라미터의 기울기 계산 비활성화 (가중치 고정) |
| `model.fc.parameters()` | 새로 추가한 FC층 파라미터만 학습 |

---

## VGG16

**3×3 합성곱 필터만을 반복 사용하는 단순하고 균일한 구조의 CNN 사전 훈련 모델**

- 2014 ILSVRC 2위 — 옥스퍼드 VGG(Visual Geometry Group)팀 개발
- 16층 (가중치 레이어 기준): Conv 13개 + FC 3개
- 모든 합성곱 레이어에 3×3 필터 사용 → 작은 수용 영역으로도 복잡한 특징 학습 가능
- Max Pooling으로 점진적으로 해상도를 줄이며 특징 추출

### 구조

| 블록 | 레이어 구성 | 출력 크기 |
| --- | --- | --- |
| Block 1 | Conv3-64 × 2, MaxPool | 112×112×64 |
| Block 2 | Conv3-128 × 2, MaxPool | 56×56×128 |
| Block 3 | Conv3-256 × 3, MaxPool | 28×28×256 |
| Block 4 | Conv3-512 × 3, MaxPool | 14×14×512 |
| Block 5 | Conv3-512 × 3, MaxPool | 7×7×512 |
| FC층 | FC-4096 × 2, FC-1000 | 1,000 (클래스 수) |

> 약 1억 3,800만 개(138M)의 파라미터를 가지는 대규모 모델 — FC층이 전체 파라미터의 약 90% 차지. 연산량과 메모리 사용량이 큰 것이 단점

### ResNet-50 vs VGG16

| 항목 | ResNet-50 | VGG16 |
| --- | --- | --- |
| 파라미터 수 | 약 2,560만 | 약 1억 3,800만 |
| 깊이 | 50층 | 16층 |
| 특징 | Skip Connection | 단순 순차 구조 |
| 메모리 사용량 | 적음 | 많음 |
| 주 활용 | 실무 / Fine Tuning | 교육 / 특징 추출 |

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 전이 학습 | ImageNet 사전 훈련 가중치 제공 — 적은 데이터로도 높은 성능 달성 |
| 단순한 구조 | 규칙적인 네트워크 구조로 이해·수정이 쉬움 — 학습 입문에 적합 |
| 범용성 | 다양한 이미지 인식·분류 작업에서 검증된 성능 |

### 코드 예시 (미니퀘스트)

#### TensorFlow (Keras)

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

num_classes = 10

x_train = np.random.random((50, 224, 224, 3)).astype(np.float32)
y_train = np.random.randint(num_classes, size=(50,))
x_test = np.random.random((10, 224, 224, 3)).astype(np.float32)
y_test = np.random.randint(num_classes, size=(10,))

# 사전 훈련된 VGG16 로드 (분류 헤드 제외)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# 분류 헤드 추가
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=8, validation_split=0.1)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")
model.summary()
```

#### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

num_classes = 10
x_train = torch.rand(50, 3, 224, 224)
y_train = torch.randint(num_classes, (50,))
x_test = torch.rand(10, 3, 224, 224)
y_test = torch.randint(num_classes, (10,))

loader = DataLoader(TensorDataset(x_train, y_train), batch_size=8)

# 사전 훈련된 VGG16 로드
model = models.vgg16(pretrained=True)

# 합성곱 레이어 가중치 고정
for param in model.features.parameters():
    param.requires_grad = False

# 분류 헤드 교체
model.classifier[6] = nn.Linear(4096, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

model.train()
for epoch in range(3):
    running_loss = 0.0
    for inputs, labels in loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(loader):.4f}')

model.eval()
with torch.no_grad():
    _, predicted = torch.max(model(x_test), 1)
    accuracy = (predicted == y_test).float().mean().item()
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

| 코드 | 설명 |
| --- | --- |
| `model.features.parameters()` | VGG16 합성곱 블록의 파라미터 (가중치 고정 대상) |
| `model.classifier[6]` | VGG16 분류기의 마지막 FC층 — 클래스 수에 맞게 교체 |
| `model.classifier.parameters()` | 교체된 분류 헤드 파라미터만 학습 |
