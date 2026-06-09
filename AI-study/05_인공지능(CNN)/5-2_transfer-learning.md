# 5-2. Transfer Learning · Fine Tuning · Feature Extraction · Zero-Shot Learning · Model Comparison · 성능 시각화

---

## 전이 학습 (Transfer Learning)

**사전 훈련된 모델을 새로운 문제에 적용하는 머신러닝 기법 — 대규모 데이터셋에서 학습한 가중치와 특성을 새 과제에 재활용하여 학습 시간과 데이터를 절약함**

### 3가지 기법

| 기법 | 설명 | 사용 시기 |
| --- | --- | --- |
| Fine Tuning | 사전 훈련 모델 일부 레이어 해동 후 재학습 | 데이터 중간 이상, 유사 도메인 |
| Feature Extraction | 모델 전체 고정, 새 분류 헤드만 학습 | 데이터 소량, 유사 도메인 |
| Zero-Shot Learning | 학습 없이 새로운 클래스 추론 | 학습 데이터가 전혀 없는 경우 |

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 데이터 절약 | 소규모 데이터셋에서도 사전 훈련 모델의 지식을 활용하여 높은 성능 달성 |
| 시간 절약 | 처음부터 학습하는 것보다 빠른 수렴 |
| 성능 향상 | 다양한 데이터에서 학습한 일반화된 특성 재사용 |
| 계산 자원 절약 | 처음부터 학습 대비 GPU 사용량·학습 비용 절감 |

---

## Fine Tuning (파인 튜닝)

**사전 훈련된 모델을 새로운 데이터셋에 맞게 미세 조정하는 과정 — 일부 또는 전체 레이어의 가중치를 재학습하여 새 도메인에 최적화**

- 하위 레이어(입력층에 가까운 레이어): 저수준 특징(엣지, 텍스처) 포착 → 범용적이므로 보통 고정
- 상위 레이어(출력층에 가까운 레이어): 고수준 특징 추출 및 분류 → 새 도메인에 맞게 재학습

### Fine Tuning 종류

| 종류 | 설명 |
| --- | --- |
| Full Fine Tuning | 모든 파라미터 재학습. 최고 성능 기대, 학습 비용 높음 |
| Partial Fine Tuning | 상위 레이어만 업데이트, 하위 레이어 고정. 빠른 학습, 과적합 방지 |
| 단계적 파인튜닝 | 상위 → 하위 레이어 순서로 점진적 해동. 학습 안정성 향상 |
| 하이브리드 | 여러 전략 조합. 상황에 맞게 유연하게 구성 |

### 코드 예시 (미니퀘스트 — Partial Fine Tuning)

#### TensorFlow (Keras)

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers

num_classes = 10
X = np.random.rand(500, 32, 32, 3).astype(np.float32)
y = np.random.randint(num_classes, size=500)

base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers[-4:]:
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax'),
])
model.compile(optimizer=optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=5, validation_split=0.2)
```

| 코드 | 설명 |
| --- | --- |
| `layer.trainable = False` | 레이어 가중치 고정 (학습 중 업데이트 안 함) |
| `base_model.layers[-4:]` | 마지막 4개 레이어만 학습 허용 (Partial Fine Tuning) |

#### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16, VGG16_Weights

num_classes = 10
model = vgg16(weights=VGG16_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
for layer in list(model.children())[-4:]:
    for param in layer.parameters():
        param.requires_grad = True
model.classifier[6] = nn.Linear(4096, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()

X = torch.rand(50, 3, 32, 32).to(device)
y = torch.randint(0, num_classes, (50,)).to(device)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=16, shuffle=True)

model.train()
for epoch in range(5):
    for inputs, labels in loader:
        optimizer.zero_grad()
        criterion(model(inputs), labels).backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/5] complete")
```

| 코드 | 설명 |
| --- | --- |
| `param.requires_grad = False` | 파라미터 업데이트 비활성화 (동결) |
| `filter(lambda p: p.requires_grad, ...)` | 학습 가능한 파라미터만 옵티마이저에 전달 |

---

## Feature Extraction (특징 추출)

**사전 훈련된 모델의 가중치를 완전히 고정하고, 새로 추가한 분류 헤드만 학습하는 전이 학습 방식**

- 중간 레이어가 이미 저수준(엣지, 텍스처)~고수준(형태, 물체) 특징을 학습한 상태를 그대로 활용
- Fine Tuning과의 차이: Fine Tuning은 일부 레이어를 해동하여 재학습, Feature Extraction은 모든 레이어 완전 고정

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 학습 부담 감소 | 학습 파라미터 수 감소 → 훈련 시간·메모리 절감 |
| 범용 특성 활용 | 사전 훈련 모델의 폭넓은 특징을 그대로 재사용 |
| 적은 데이터로 높은 성능 | 과소적합 위험 감소, 소규모 데이터셋에서도 효과적 |
| 빠른 프로토타이핑 | 구현 난이도가 낮아 다양한 모델 조합을 빠르게 실험 가능 |

### 코드 예시 (미니퀘스트)

#### TensorFlow (Keras)

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

num_classes = 10
base_model = tf.keras.applications.ResNet50(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg'
)
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    base_model,
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

X_train = np.random.rand(100, 224, 224, 3).astype(np.float32)
y_train = np.random.randint(0, num_classes, 100)
model.fit(X_train, y_train, epochs=3, batch_size=10)
```

| 코드 | 설명 |
| --- | --- |
| `pooling='avg'` | Global Average Pooling 적용 — Flatten 없이 분류 헤드로 바로 연결 |
| `base_model.trainable = False` | 모든 레이어 완전 고정 (Fine Tuning과의 핵심 차이) |

#### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10

model = resnet50(weights=ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128), nn.ReLU(), nn.Linear(128, num_classes)
)
model = model.to(device)

optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

X = torch.rand(50, 3, 224, 224).to(device)
y = torch.randint(0, num_classes, (50,)).to(device)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=10)

model.train()
for epoch in range(3):
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        criterion(model(x_batch), y_batch).backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/3] complete")
```

| 코드 | 설명 |
| --- | --- |
| `for param in model.parameters(): param.requires_grad = False` | 모든 파라미터 동결 |
| `optimizer = optim.Adam(model.fc.parameters(), ...)` | 새로 추가한 FC층 파라미터만 학습 |

---

## Zero-Shot Learning (제로샷 학습)

**훈련 데이터에 없는 새로운 클래스를 사전 학습된 개념 관계를 이용해 학습 없이 추론하는 기법**

- 텍스트 임베딩, 속성 벡터, 멀티모달 표현(이미지+텍스트)을 활용
- CLIP(OpenAI)은 이미지와 텍스트를 같은 임베딩 공간에 매핑하여 제로샷 분류 지원

### 동작 방식

| 단계 | 설명 |
| --- | --- |
| 1 | 사전 훈련 모델이 이미지와 텍스트를 동일한 임베딩 공간에 매핑 |
| 2 | 새 클래스의 텍스트 설명을 임베딩으로 변환 |
| 3 | 이미지 임베딩과 각 클래스 텍스트 임베딩 간의 유사도 계산 |
| 4 | 유사도가 가장 높은 클래스를 예측 결과로 반환 |

### Zero-Shot vs Few-Shot vs Fine Tuning

| 항목 | Zero-Shot | Few-Shot | Fine Tuning |
| --- | --- | --- | --- |
| 훈련 샘플 수 | 0 | 1~수십 장 | 수백 장 이상 |
| 추가 학습 | 불필요 | 최소 학습 | 필요 |
| 적용 범위 | 광범위 | 제한적 | 특정 도메인 |

### 코드 예시 (미니퀘스트 — CLIP)

```python
# pip install transformers
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.new('RGB', (224, 224), color=(128, 128, 128))
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car", "a photo of a zebra"]

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)

probs = outputs.logits_per_image.softmax(dim=1)
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob.item():.4f}")
```

| 코드 | 설명 |
| --- | --- |
| `CLIPModel` | 이미지와 텍스트를 같은 임베딩 공간에 매핑하는 멀티모달 모델 |
| `logits_per_image` | 이미지와 각 텍스트 간의 유사도 점수 |
| `.softmax(dim=1)` | 유사도를 확률로 변환 |

---

## 모델 비교 (Model Comparison)

**여러 모델을 동일한 데이터셋으로 훈련·평가하여 가장 효과적인 모델을 선택하는 과정**

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 최적 성능 확보 | 문제를 가장 잘 해결할 수 있는 모델 선택 |
| 모델 적합성 확인 | 특정 데이터셋에 어떤 구조가 적합한지 파악 |
| 효율적인 자원 사용 | 성능 대비 계산 자원·학습 시간 비교 가능 |
| 과적합/과소적합 진단 | 학습·검증 성능 차이를 통해 모델 상태 파악 |

### 코드 예시 (미니퀘스트)

#### TensorFlow (Keras)

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

num_classes = 10
X = np.random.random((1200, 32, 32, 3)).astype(np.float32)
y = to_categorical(np.random.randint(num_classes, size=(1200, 1)), num_classes)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_simple_cnn():
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)), layers.Flatten(),
        layers.Dense(128, activation='relu'), layers.Dense(num_classes, activation='softmax'),
    ])

def build_deep_cnn():
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)), layers.Flatten(),
        layers.Dense(256, activation='relu'), layers.Dense(num_classes, activation='softmax'),
    ])

results = {}
for name, build_fn in [('SimpleCNN', build_simple_cnn), ('DeepCNN', build_deep_cnn)]:
    m = build_fn()
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    m.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
    _, acc = m.evaluate(x_test, y_test, verbose=0)
    results[name] = acc
    print(f"{name} — Accuracy: {acc:.4f}")

print(f"Best model: {max(results, key=results.get)}")
```

#### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

num_classes = 10
X = torch.tensor(np.random.random((1200, 3, 32, 32)).astype(np.float32))
y = torch.tensor(np.random.randint(num_classes, size=(1200,)))
train_loader = DataLoader(TensorDataset(X[:960], y[:960]), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X[960:], y[960:]), batch_size=32)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32*16*16, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(128*4*4, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.net(x)

def train_eval(model, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
    model.eval()
    correct = sum((model(xb).argmax(1) == yb).sum().item() for xb, yb in test_loader)
    return correct / len(test_loader.dataset)

for name, cls in [('SimpleCNN', SimpleCNN), ('DeepCNN', DeepCNN)]:
    acc = train_eval(cls())
    print(f"{name} — Accuracy: {acc:.4f}")
```

---

## 성능 시각화 (Performance Visualization)

**모델의 학습 과정과 최종 성능 지표(정확도, 손실 등)를 그래프로 표현하여 직관적으로 파악하는 방법**

- 학습 곡선을 통해 과적합·과소적합 여부를 시각적으로 확인 가능
- `matplotlib.pyplot`(plt)을 사용하여 그래프 출력

### 시각화 유형

| 유형 | 내용 | 활용 |
| --- | --- | --- |
| Loss / Accuracy 곡선 | 에포크별 학습·검증 손실 및 정확도 추이 | 과적합·과소적합 진단 |
| Confusion Matrix | 클래스별 예측 결과를 행렬로 표현 | 오분류 패턴 확인 |
| TensorBoard | 학습 과정을 실시간으로 모니터링 | 학습 중 상태 추적 |

### 과적합 / 과소적합 진단

| 상태 | 징후 |
| --- | --- |
| 과적합 (Overfitting) | 학습 정확도↑ + 검증 정확도 정체 / 학습 손실↓ + 검증 손실↑ |
| 과소적합 (Underfitting) | 학습·검증 정확도 모두 낮음 / 에포크가 늘어도 성능 개선 미미 |

### 코드 예시 (미니퀘스트)

#### TensorFlow (Keras)

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

num_classes = 10
X = np.random.random((1200, 32, 32, 3)).astype(np.float32)
y = to_categorical(np.random.randint(num_classes, size=(1200, 1)), num_classes)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)), layers.Flatten(),
    layers.Dense(128, activation='relu'), layers.Dense(num_classes, activation='softmax'),
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()
plt.tight_layout()
plt.show()
```

#### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

num_classes = 10
X = torch.tensor(np.random.random((1200, 3, 32, 32)).astype(np.float32))
y = torch.tensor(np.random.randint(num_classes, size=(1200,)))
train_loader = DataLoader(TensorDataset(X[:960], y[:960]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X[960:], y[960:]), batch_size=32)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32*16*16, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_accs, val_accs, train_losses, val_losses = [], [], [], []

for epoch in range(10):
    model.train()
    t_loss, t_correct, t_total = 0, 0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        t_correct += (out.argmax(1) == yb).sum().item()
        t_total += yb.size(0)

    model.eval()
    v_loss, v_correct, v_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)
            v_loss += criterion(out, yb).item()
            v_correct += (out.argmax(1) == yb).sum().item()
            v_total += yb.size(0)

    train_losses.append(t_loss / len(train_loader))
    val_losses.append(v_loss / len(val_loader))
    train_accs.append(t_correct / t_total)
    val_accs.append(v_correct / v_total)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_accs, label='Train')
axes[0].plot(val_accs, label='Validation')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[1].plot(train_losses, label='Train')
axes[1].plot(val_losses, label='Validation')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()
plt.tight_layout()
plt.show()
```

| 코드 | 설명 |
| --- | --- |
| `history.history` | 에포크별 loss, val_loss, accuracy, val_accuracy 기록 딕셔너리 |
| `plt.subplots(1, 2, ...)` | 1행 2열 서브플롯 생성 — 정확도와 손실 나란히 표시 |
| `val_loader` | 학습에 사용하지 않은 데이터로 일반화 성능 평가 |
