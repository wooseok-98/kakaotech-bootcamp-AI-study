# 6-1. Overfitting · Underfitting · Early Stopping · Hyperparameter Tuning

---

## 과적합 (Overfitting)

**학습 데이터를 과하게 학습하여 새로운 데이터에 대한 오차가 증가하는 현상**

- 학습 데이터의 노이즈나 특이 패턴까지 학습하여 일반적인 패턴을 놓침
- 학습 정확도↑ + 검증 정확도↓ 또는 정체 형태로 나타남

### 발생 이유

| 이유 | 설명 |
| --- | --- |
| 모델 복잡성 | 파라미터 수가 너무 많거나 복잡한 구조의 모델은 학습 데이터에 과적합되기 쉬움 |
| 데이터 양 부족 | 학습 데이터가 적으면 모델이 데이터를 외워버림 |
| 노이즈 포함 학습 | 데이터에 노이즈가 많으면 모델이 노이즈까지 학습하여 새로운 데이터 성능 저하 |

### 방지 방법

| 방법 | 설명 |
| --- | --- |
| Dropout | 학습 중 무작위로 뉴런을 비활성화하여 특정 뉴런에 의존하지 않도록 함 |
| 정규화 (L1/L2) | 큰 가중치 값에 패널티를 부여하여 모델 복잡성을 제한 |
| 데이터 증강 | 학습 데이터를 다양하게 변환하여 데이터량 증가 |
| Early Stopping | 검증 손실이 더 이상 개선되지 않으면 학습 조기 종료 |
| 교차 검증 | 데이터를 여러 번 나누어 검증하여 일반화 성능 평가 |
| 앙상블 | 여러 모델을 결합하여 예측 안정성 향상 |

### 코드 예시 (미니퀘스트 — Dropout + Early Stopping)

#### TensorFlow (Keras)

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

X = np.random.rand(100, 2)
y = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(100).batch(10)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(10)

model = models.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2),
])
model.compile(
    optimizer=optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)
model.fit(train_ds, validation_data=test_ds, epochs=100, callbacks=[early_stopping], verbose=0)
_, acc = model.evaluate(test_ds, verbose=0)
print(f"Test Accuracy: {acc:.4f}")
```

| 코드 | 설명 |
| --- | --- |
| `Dropout(0.5)` | 학습 시 50% 확률로 뉴런 비활성화 — 특정 뉴런 의존도 감소 |
| `EarlyStopping(monitor='val_loss', patience=10)` | 10 에포크 동안 val_loss 개선 없으면 학습 중단 |
| `restore_best_weights=True` | 조기 종료 시 가장 낮은 val_loss 시점의 가중치 복원 |

#### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)
torch.manual_seed(42)

X = np.random.rand(100, 2).astype(np.float32)
y = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)),
    batch_size=10, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long)),
    batch_size=10
)

class SimpleNNWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 2),
        )
    def forward(self, x): return self.net(x)

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), 'checkpoint.pt')
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

model = SimpleNNWithDropout()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
es = EarlyStopping(patience=10)

for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            val_loss += criterion(model(xb), yb).item()
    val_loss /= len(test_loader)

    es(val_loss, model)
    if es.early_stop:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(torch.load('checkpoint.pt'))
```

| 코드 | 설명 |
| --- | --- |
| `nn.Dropout(0.5)` | 학습 시 50% 뉴런 비활성화 (평가 모드에서 자동으로 비활성화됨) |
| `torch.save(model.state_dict(), 'checkpoint.pt')` | 검증 손실 개선 시 최적 가중치 저장 |
| `model.load_state_dict(torch.load('checkpoint.pt'))` | 학습 종료 후 저장된 최적 가중치 복원 |

---

## 과소적합 (Underfitting)

**모델이 데이터의 복잡성을 충분히 학습하지 못해 학습 데이터와 새로운 데이터 모두에서 낮은 성능을 보이는 현상**

- 과적합과 달리 학습 데이터에서도 성능이 낮음
- 모델이 지나치게 단순하거나 충분히 학습되지 않은 경우 발생

### 특징

| 특징 | 설명 |
| --- | --- |
| 학습 데이터 성능 자체가 낮음 | 간단한 패턴도 놓치고 있어 학습 데이터조차 제대로 맞추지 못함 |
| 새로운 데이터 성능도 낮음 | 학습 데이터 성능이 낮으면 테스트 데이터에서도 성능이 낮아짐 |
| 지나치게 단순한 모델 | 복잡한 패턴을 학습할 능력이 부족하거나 충분히 훈련되지 않은 상태 |

### 방지 방법

| 카테고리 | 방법 | 설명 |
| --- | --- | --- |
| 모델 구조 개선 | 레이어/뉴런 수 증가 | 더 많은 레이어 추가로 표현력 향상 |
|  | 활성화 함수 변경 | sigmoid → ReLU로 변경하여 기울기 소실 방지 |
|  | 적절한 아키텍처 선택 | 문제에 맞는 사전 훈련 모델(ResNet, VGG 등) 선택 |
| 데이터 활용 최적화 | 더 많은 데이터 확보 | 다양한 데이터로 모델의 일반화 향상 |
|  | 데이터 증강 | 회전, 반전 등 다양한 변형으로 데이터 다양성 확보 |
|  | 데이터 전처리 개선 | 정규화, 노이즈 제거, 불균형 데이터 처리 |
| 학습 설정 조정 | 학습률 조정 | Adam: 0.0001~0.001 / SGD: 0.01~0.1 범위에서 조정 |
|  | Epoch 증가 | 소규모 데이터셋: 10~50 / 중간 규모: 50~200 |
|  | 최적화 알고리즘 변경 | SGD → Adam 등 적합한 옵티마이저 선택 |

### 코드 예시 (Underfitting 개선)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

num_classes = 10
input_shape = (32, 32, 3)

# 개선 전: 얕은 구조 + sigmoid 활성화 → Underfitting 발생 가능
model_before = Sequential([
    Input(shape=input_shape),
    Conv2D(32, (3, 3), activation='sigmoid'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='sigmoid'),
    Dense(num_classes, activation='softmax'),
])

# 개선 후: 깊은 구조 + relu 활성화
model_after = Sequential([
    Input(shape=input_shape),
    Conv2D(32, (3, 3), activation='relu'),
    Conv2D(32, (3, 3), activation='relu'),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax'),
])
```

| 변경 사항 | 이유 |
| --- | --- |
| Conv2D 1개 → 3개 | 더 많은 특징 학습으로 표현력 향상 |
| Dense 1개 → 3개 | 분류 능력 강화 |
| `sigmoid` → `relu` | 기울기 소실 문제 해결, 학습 안정성 향상 |

---

## 조기 종료 (Early Stopping)

**학습 과정에서 검증 손실이 더 이상 개선되지 않는 시점을 감지하여 과적합을 방지하기 위해 학습을 조기에 중단하는 기법**

- 검증 데이터의 성능 지표(val_loss, val_accuracy)를 에포크마다 모니터링
- `patience` 에포크 동안 개선이 없으면 학습 중단 → `restore_best_weights`로 최적 가중치 복원

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 과적합 방지 | 검증 손실이 증가하기 시작하는 시점에 학습을 멈춰 과적합 방지 |
| 훈련 시간 절약 | 성능 개선이 없는 추가 에포크를 건너뛰어 시간·자원 절감 |
| 최적 모델 선택 | 검증 성능이 가장 높은 지점의 가중치를 자동으로 선택 |

### 조기 종료 조건

| 모니터링 대상 | 설명 |
| --- | --- |
| `val_loss` | 검증 손실이 감소하지 않으면 중단 (가장 일반적) |
| `val_accuracy` | 검증 정확도가 증가하지 않으면 중단 |
| 사용자 정의 조건 | 손실 증가율 등 모델 특성에 맞게 커스터마이징 가능 |

### 코드 예시 (미니퀘스트)

#### TensorFlow (Keras)

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

X_train = np.random.rand(1000, 20)
y_train = np.random.randint(2, size=1000)
X_val = np.random.rand(200, 20)
y_val = np.random.randint(2, size=200)

model = Sequential([
    Input(shape=(20,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    X_train, y_train, epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=0
)
print(f"Stopped at epoch {len(history.history['val_loss'])}")
_, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Val Accuracy: {acc:.4f}")
```

| 코드 | 설명 |
| --- | --- |
| `EarlyStopping(monitor='val_loss', patience=3)` | val_loss가 3 에포크 동안 개선 없으면 학습 중단 |
| `restore_best_weights=True` | 학습 중 가장 낮은 val_loss 시점의 가중치 복원 |

#### PyTorch

TF와 달리 내장 EarlyStopping이 없어 직접 구현해야 함

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

X_train = torch.tensor(np.random.rand(1000, 20).astype(np.float32))
y_train = torch.tensor(np.random.randint(2, size=1000).astype(np.float32))
X_val = torch.tensor(np.random.rand(200, 20).astype(np.float32))
y_val = torch.tensor(np.random.randint(2, size=200).astype(np.float32))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

patience, counter, best_loss = 3, 0, float('inf')

for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        criterion(model(xb).squeeze(), yb).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = sum(
            criterion(model(xb).squeeze(), yb).item() for xb, yb in val_loader
        ) / len(val_loader)

    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(torch.load('best_model.pt'))
```

| 코드 | 설명 |
| --- | --- |
| `patience, counter, best_loss` | 조기 종료를 위한 변수 (인내 에포크 수, 현재 카운터, 최소 손실) |
| `torch.save(model.state_dict(), ...)` | 검증 손실 개선 시 최적 가중치 저장 |
| `model.load_state_dict(...)` | 학습 종료 후 저장된 최적 가중치 복원 |

---

## 하이퍼파라미터 튜닝 (Hyperparameter Tuning)

**머신러닝 모델의 성능을 최적화하기 위해 모델의 하이퍼파라미터 값을 조정하는 과정**

- 하이퍼파라미터: 학습 과정에 영향을 미치지만 학습으로 자동 결정되지 않아 직접 설정해야 하는 값
- 다양한 조합을 탐색하고 평가하여 모델 성능이 좋아지도록 최적화

### 주요 하이퍼파라미터

| 이름 | 설명 | 예시 |
| --- | --- | --- |
| 학습률 (Learning Rate) | 가중치 업데이트의 크기를 결정 | 0.01, 0.001, 0.0001 |
| 배치 크기 (Batch Size) | 한 번에 학습할 데이터 샘플 수 | 16, 32, 64, 128 |
| 에포크 수 (Epochs) | 전체 데이터셋 반복 횟수 | 10, 50, 100 |
| 옵티마이저 (Optimizer) | 모델 학습을 최적화하는 알고리즘 | SGD, Adam, RMSprop |
| 드롭아웃 비율 (Dropout Rate) | 학습 중 무작위로 끄는 뉴런 비율 | 0.2, 0.5 |
| 정규화 파라미터 | 과적합 방지를 위한 가중치 패널티 값 | L2 lambda: 0.01, 0.001 |
| 활성화 함수 | 각 뉴런 출력값을 결정하는 함수 | ReLU, Sigmoid, Tanh |

### 튜닝 방법

| 방법 | 설명 | 특징 |
| --- | --- | --- |
| Grid Search | 설정한 파라미터 조합을 모두 탐색 | 확실하지만 계산 비용 높음 |
| Random Search | 무작위로 파라미터 조합을 선택하여 탐색 | 빠르고 효율적, 최적값 보장 없음 |

### 코드 예시 (미니퀘스트 — Grid Search)

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

num_classes = 10
input_shape = (32, 32, 3)

X = np.random.rand(1000, 32, 32, 3)
y = np.random.randint(num_classes, size=1000)
X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_model(learning_rate=0.001):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]
best_accuracy, best_params = 0, {}

for lr in learning_rates:
    for batch_size in batch_sizes:
        model = create_model(learning_rate=lr)
        model.fit(X_train, y_train, epochs=10, batch_size=batch_size, validation_split=0.2, verbose=0)
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'learning_rate': lr, 'batch_size': batch_size}

best_model = create_model(learning_rate=best_params['learning_rate'])
best_model.fit(X_train, y_train, epochs=10, batch_size=best_params['batch_size'], validation_split=0.2)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"Best Params: {best_params}, Test Accuracy: {test_accuracy:.2f}")
```

| 코드 | 설명 |
| --- | --- |
| `learning_rates`, `batch_sizes` | 탐색할 하이퍼파라미터 후보값 리스트 |
| 이중 for 루프 | 모든 파라미터 조합을 순차적으로 탐색 (Grid Search) |
| `best_params` | 가장 높은 검증 정확도를 낸 파라미터 조합 저장 |

### 코드 예시 (미니퀘스트 — Random Search)

```python
import numpy as np
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.utils import shuffle
from scipy.stats import uniform
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

num_classes = 10
input_shape = (32, 32, 3)

X = np.random.rand(1000, 32, 32, 3)
y = np.random.randint(num_classes, size=1000)
X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_model(learning_rate=0.001):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

param_dist = {
    'learning_rate': uniform(0.001, 0.1),
    'batch_size': [16, 32, 64],
}
best_accuracy, best_params = 0, {}

for params in ParameterSampler(param_dist, n_iter=5, random_state=42):
    model = create_model(learning_rate=params['learning_rate'])
    model.fit(X_train, y_train, epochs=10, batch_size=params['batch_size'], validation_split=0.2, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

best_model = create_model(learning_rate=best_params['learning_rate'])
best_model.fit(X_train, y_train, epochs=10, batch_size=best_params['batch_size'], validation_split=0.2)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"Best Params: {best_params}, Test Accuracy: {test_accuracy:.2f}")
```

| 코드 | 설명 |
| --- | --- |
| `uniform(0.001, 0.1)` | 0.001~0.1 사이 균일 분포에서 무작위 샘플링 |
| `ParameterSampler(param_dist, n_iter=5)` | n_iter 횟수만큼 무작위 파라미터 조합 생성 |

### GridSearchCV · RandomSearchCV (scikit-learn)

scikit-learn 모델(Random Forest 등)에서는 `GridSearchCV`, `RandomizedSearchCV`를 사용하여 교차 검증까지 자동으로 수행할 수 있음

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from scipy.stats import randint

X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)

# GridSearchCV — 모든 조합 탐색 + 5-fold 교차 검증
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("GridSearch Best:", grid_search.best_params_)
best = grid_search.best_estimator_
print(f"Train: {best.score(X_train, y_train):.4f}, Test: {best.score(X_test, y_test):.4f}")

# RandomizedSearchCV — 무작위 조합 100개 탐색 + 5-fold 교차 검증
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(10, 30),
    'min_samples_split': randint(2, 10),
}
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)
print("RandomSearch Best:", random_search.best_params_)
best = random_search.best_estimator_
print(f"Train: {best.score(X_train, y_train):.4f}, Test: {best.score(X_test, y_test):.4f}")
```

| 코드 | 설명 |
| --- | --- |
| `GridSearchCV(cv=5, scoring='accuracy')` | 5-fold 교차 검증으로 모든 파라미터 조합 평가 |
| `RandomizedSearchCV(n_iter=100)` | 100개의 무작위 조합만 선택하여 계산 비용 절감 |
| `best_estimator_` | 최적 파라미터로 학습된 모델 객체 반환 |
