# 3-2. ML 알고리즘 — Random Forest · K-NN · SVM · Naive Bayes

---

## Random Forest (랜덤 포레스트)

**의사결정 트리(Decision Tree) 기반의 앙상블 학습 기법으로, 다수의 트리를 결합하여 예측 성능을 향상시키는 모델**

| 용어 | 의미 |
| --- | --- |
| Random | 예측 불가능하고 패턴이 없는 무작위성 |
| Forest | 여러 개의 트리(Tree)로 구성된 집합 |
| Random Forest | 무작위로 선택된 데이터·특성으로 학습한 여러 의사결정 트리를 앙상블로 결합한 모델 |

### 의사결정 트리 (Decision Tree)

**데이터의 특성(feature)을 기준으로 조건 분기하여 예측 결과를 도출하는 트리 구조 모델**

```
              [오늘 비가 오나요?]
                 /          \
               예            아니오
              /                  \
  [바람이 강한가요?]          [우산 필요 없음]
     /        \
   예          아니오
  /               \
[우산 필요]    [우산 필요]
```

| 구성 요소 | 설명 |
| --- | --- |
| 루트 노드 (Root Node) | 트리의 시작점 — 전체 데이터셋 포함 |
| 내부 노드 (Internal Node) | 특성 값을 기준으로 데이터를 분할하는 노드 |
| 리프 노드 (Leaf Node) | 최종 예측값을 나타내는 노드 — 더 이상 분할되지 않음 |

### 앙상블 기법 (Ensemble Method)

**여러 개의 예측 모델을 결합하여 단일 모델보다 더 높은 예측 성능을 얻는 기법**

| 기법 | 설명 | 대표 모델 |
| --- | --- | --- |
| 배깅 (Bagging) | 데이터를 중복 허용 무작위 샘플링 후 모델을 독립적으로 학습 → 다수결/평균으로 결합 | **Random Forest** |
| 부스팅 (Boosting) | 이전 모델이 틀린 부분을 보완하며 순차적으로 학습 → 오류에 가중치를 부여 | Gradient Boosting |
| 스태킹 (Stacking) | 서로 다른 알고리즘의 예측값을 메타 모델의 입력으로 사용 | 다양한 조합 |

> **Random Forest = 배깅 기반 앙상블 모델**
> 각 트리는 중복 허용 무작위 샘플 + 무작위 특성 부분집합으로 학습 → 트리 간 상관관계를 낮춰 일반화 성능 향상

### 동작 방식

| 동작 | 설명 |
| --- | --- |
| 부트스트랩 샘플링 | 전체 훈련 데이터에서 중복을 허용하여 무작위로 샘플 선택 — 각 트리마다 다른 데이터로 학습 |
| 다수의 트리 생성 | 각 노드에서 일부 특성만 무작위 선택 후 최적 분할 → 트리 간 상관관계 감소 |
| 예측 결합 | 분류: 다수결 투표 / 회귀: 평균 → 최종 예측값 결정 |

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 과적합 방지 | 여러 트리의 예측을 조합하여 단일 트리의 과적합 문제를 완화 |
| 노이즈·이상치에 강함 | 여러 트리의 결과를 종합하여 안정적인 예측 성능 유지 |
| 특성 중요도 제공 | 각 특성이 예측에 미치는 영향(Feature Importance)을 수치로 확인 가능 |
| 분류·회귀 모두 가능 | 범주형(분류)과 연속형(회귀) 문제 모두에 적용 가능 |
| 병렬 처리 가능 | 트리를 독립적으로 학습하므로 대규모 데이터셋에서도 효율적 |

### 코드 예시

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 데이터 로드 (펭귄 데이터셋)
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins.csv"
df = pd.read_csv(dataset_url).dropna()

# 레이블 인코딩 (Adelie: 0, Chinstrap: 1, Gentoo: 2)
label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])

# 원-핫 인코딩 및 데이터 분할
X = pd.get_dummies(df.drop(columns=["species"])).astype(float)
y = df["species"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 기본 모델 (트리 100개)
rf_model1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model1.fit(X_train, y_train)

# 하이퍼파라미터 튜닝 모델 (트리 100개, 최대 깊이 10)
rf_model2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model2.fit(X_train, y_train)

# 평가
print(f"기본 모델 정확도:  {rf_model1.score(X_test, y_test):.4f}")
print(f"튜닝 모델 정확도:  {rf_model2.score(X_test, y_test):.4f}")

# 예측 확률 (상위 5개)
print(rf_model1.predict_proba(X_test)[:5])
```

| 파라미터 | 설명 |
| --- | --- |
| `n_estimators` | 생성할 결정 트리의 수 — 클수록 안정적이지만 학습 시간 증가 |
| `max_depth` | 각 트리의 최대 깊이 — 제한할수록 과적합 방지 |
| `predict_proba()` | 각 클래스에 대한 예측 확률 반환 |
| `.flatten()` | 다차원 배열을 1차원 배열로 변환 |

---

## K-NN (K-Nearest Neighbors, K-최근접 이웃)

**새로운 데이터를 가장 가까운 K개의 이웃과 비교하여 분류하거나 예측하는 알고리즘**

| 용어 | 의미 |
| --- | --- |
| K | 분류하려는 데이터 포인트와 가장 가까운 이웃 데이터 포인트의 개수 |
| Nearest Neighbors | 거리가 가장 가까운 이웃 데이터들 |
| K-NN | K개의 이웃 라벨을 기준으로 새로운 데이터를 분류·예측하는 알고리즘 |

### 동작 방식

학습 단계에서는 별도의 파라미터를 추정하지 않고 **전체 훈련 데이터를 저장**하기만 함
→ 예측 시점에 거리를 계산하여 이웃을 찾기 때문에 **메모리 기반 학습(Lazy Learning)** 이라고 함

1. 입력 데이터(예측 대상) 입력
2. 저장된 훈련 데이터 전체와의 거리 계산
3. 거리가 가장 가까운 K개의 데이터를 이웃으로 선택
4. 분류: 이웃의 다수결 라벨 / 회귀: 이웃의 평균값 → 최종 결과 산출

### K 값의 영향

| K 값 | 특성 |
| --- | --- |
| 너무 작음 | 노이즈에 민감, 특정 데이터에 과도하게 의존 → 과적합 위험 |
| 너무 큼 | 세밀한 차이를 놓침 → 과소적합 발생 |
| 적절한 값 | 데이터셋과 문제 특성에 따라 실험적으로 결정 (일반적으로 홀수 권장) |

### 거리 척도

| 척도 | 설명 | 수식 |
| --- | --- | --- |
| 유클리디안 거리 | 두 점 사이의 직선 거리 — 가장 일반적 | `√(Σ(xᵢ - yᵢ)²)` |
| 맨해튼 거리 | 각 좌표축을 따라 이동하는 거리 합 — 도시 블록 거리 | `Σ|xᵢ - yᵢ|` |

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 구조가 단순하고 직관적 | 학습 단계에서 파라미터를 학습하지 않아 이해하기 쉬움 |
| 구현이 쉬움 | K값과 거리 척도만 설정하면 즉시 사용 가능 |
| 분류·회귀 모두 적용 가능 | 다수결(분류) 또는 평균(회귀)으로 동일한 원리 사용 |
| 비선형 결정 경계 처리 | 선형 모델과 달리 복잡하게 분포된 데이터도 유연하게 처리 |
| 예측 결과 해석이 쉬움 | 어떤 이웃들이 선택되었는지 직접 확인 가능 |

### 코드 예시

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 정규화 — K-NN은 거리 기반이므로 스케일 통일이 중요
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 훈련 데이터로 평균·분산 학습 후 변환
X_test  = scaler.transform(X_test)       # 동일한 기준으로 테스트 데이터 변환

# K-NN 모델 생성 및 학습 (K=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 평가
print(f"테스트 정확도: {knn.score(X_test, y_test):.4f}")

# 단일 샘플 예측
sample = X_test[0].reshape(1, -1)  # (1, n_features) 형태로 변환
prediction = knn.predict(sample)
print(f"예측된 클래스: {iris.target_names[prediction[0]]}")
```

> **K-NN에서 정규화가 중요한 이유**: 특성마다 단위가 다르면(예: 키cm vs 몸무게kg) 단위가 큰 특성이 거리를 지배함
> → `StandardScaler`로 모든 특성을 동일한 척도(평균 0, 표준편차 1)로 맞춰야 공정한 거리 계산 가능

---

## SVM (Support Vector Machine, 서포트 벡터 머신)

**주어진 데이터를 최적으로 분리하는 초평면(Hyperplane)을 찾는 지도 학습 알고리즘**

| 용어 | 의미 |
| --- | --- |
| Support Vector | 결정 경계를 정의하는 데 핵심적인 역할을 하는 데이터 샘플 |
| Hyperplane | 두 개 이상의 클래스를 구분하는 선(2D) 또는 평면(3D 이상) |
| SVM | 마진을 최대화하는 초평면을 찾아 데이터를 분류하는 지도 학습 알고리즘 |

### 핵심 개념

**초평면 (Hyperplane)**

두 클래스를 구분하는 결정 경계 — 차원에 따라 형태가 달라짐

| 차원 | 초평면 형태 |
| --- | --- |
| 2차원 | 직선 |
| 3차원 | 평면 |
| N차원 | N-1차원 초평면 |

**마진 (Margin)**

초평면과 가장 가까운 데이터 샘플(서포트 벡터) 사이의 거리
- SVM은 이 **마진을 최대화**하는 초평면을 찾는 것이 목표
- 마진이 클수록 새로운 데이터에 안정적으로 분류 가능 → 일반화 성능 향상
- 마진이 작으면 결정 경계가 특정 샘플에 과도하게 의존 → 과적합 위험 증가

**서포트 벡터 (Support Vectors)**

초평면과 가장 가까이 있는 데이터 샘플 — **초평면의 위치와 방향을 직접 결정**
- 나머지 데이터는 결정 경계 형성에 관여하지 않음
- 이 덕분에 이상치(outlier)에도 비교적 강건함

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 마진 최대화로 일반화 성능 향상 | 서포트 벡터 간 거리를 최대화하여 새로운 데이터에도 정확한 예측 가능 |
| 고차원 데이터에서 안정적 | 차원이 늘어나도 최적의 결정 경계를 유지 (K-NN 등 거리 기반 알고리즘보다 유리) |
| 소규모 데이터에서도 효과적 | 대량의 데이터를 요구하는 딥러닝과 달리 적은 데이터에서도 좋은 성능 |
| 이상치에 강함 | 전체 데이터가 아닌 서포트 벡터만으로 결정 경계를 정의하므로 이상치 영향 최소화 |

### 커널 트릭 (Kernel Trick)

선형으로 분리되지 않는 데이터를 **고차원으로 변환하여 선형 분리 가능하게 만드는 기법**

| 커널 종류 | 설명 | 특징 |
| --- | --- | --- |
| Linear | 선형 초평면으로 분류 | 계산량 적음, 선형 분리 가능한 데이터에 적합 |
| Polynomial | 다항식 변환으로 곡선 경계 형성 | 차수가 높을수록 복잡한 패턴 학습 가능하나 계산량 증가 |
| RBF (Gaussian) | 유클리디안 거리 기반 비선형 변환 | 가장 유연한 경계, SVM의 기본 커널 |
| Sigmoid | tanh 함수와 유사한 변환 | 특정 상황에서 유용하나 일반적으로 RBF보다 성능 낮음 |

### 코드 예시

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 데이터 준비
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# 정규화 — SVM은 거리 기반이므로 스케일 통일 필수
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# SVM 모델 생성 및 학습 (선형 커널)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# 평가
print(f"테스트 정확도: {svm_model.score(X_test, y_test):.4f}")

# 단일 샘플 예측
sample = X_test[0].reshape(1, -1)  # (1, n_features) 형태로 변환
prediction = svm_model.predict(sample)
print(f"예측된 클래스: {iris.target_names[prediction[0]]}")
```

> **SVM에서 정규화가 필수인 이유**: 마진과 서포트 벡터를 거리로 계산하므로 특성 간 스케일이 다르면 특정 특성이 결정 경계를 지배함
> PyTorch 환경에서는 `sklearn.svm.SVC`를 사용하고 `.numpy()`로 텐서를 변환해서 학습

---

## Naive Bayes (나이브 베이즈)

**모든 특성이 서로 독립이라고 가정하고 베이즈 정리를 활용하여 확률을 계산하는 분류 모델**

| 용어 | 의미 |
| --- | --- |
| Naive | 모든 특성이 서로 독립적이라고 가정하는 단순한 접근 방식 |
| Bayes | 베이즈 정리(Bayes' Theorem) — 조건부 확률을 계산하는 공식 |
| Naive Bayes | 독립 가정 + 베이즈 정리로 각 클래스에 속할 확률을 계산하는 분류 모델 |

### 베이즈 정리 (Bayes' Theorem)

**새로운 정보가 주어졌을 때 기존 믿음(확률)을 업데이트하는 공식**

```
P(A|B) = P(B|A) × P(A) / P(B)
```

| 수식 | 개념 | 설명 |
| --- | --- | --- |
| P(A\|B) | 사후 확률 (Posterior) | 증거 B가 주어졌을 때 A가 발생할 확률 |
| P(B\|A) | 우도 (Likelihood) | A가 발생했을 때 B가 관측될 확률 |
| P(A) | 사전 확률 (Prior) | 사전 정보에서 알고 있는 A의 확률 |
| P(B) | 정규화 상수 (Marginal) | 모든 가능한 A에 대해 B가 발생할 전체 확률 |

### 동작 방식

1. 입력 데이터가 주어지면 각 특성 값을 기반으로 **각 클래스에 속할 확률 계산**
2. 모든 클래스에 대해 확률 비교
3. **가장 높은 확률을 가지는 클래스를 최종 분류 결과로 결정**

> **독립 가정**: 특성 간 완전한 독립이 현실에서는 드물지만, 이 단순한 가정 덕분에 계산이 간단해짐
> 실제로도 독립이 완벽하지 않은 경우에도 좋은 성능을 보이는 경우가 많음

### 가우시안 나이브 베이즈 (GaussianNB)

**연속형(수치형) 데이터를 처리할 때 사용되는 나이브 베이즈 모델**

각 특성이 **정규 분포(가우시안 분포)를 따른다고 가정**하고, 각 클래스의 평균(μ)과 분산(σ²)을 이용해 확률 추정

### 사용 이유

| 이유 | 설명 |
| --- | --- |
| 계산이 빠르고 구현이 단순 | 독립 가정으로 확률 계산이 단순화 — 복잡한 파라미터 최적화 없이 각 특성의 조건부 확률만 추정 |
| 적은 데이터에서도 안정적 | 베이즈 정리 기반이라 훈련 데이터가 많지 않아도 괜찮은 성능 발휘 |
| 고차원·희소 데이터에 강함 | 텍스트 분류·스팸 필터링처럼 특성이 많고 희소한 데이터에 적합 |
| 과적합에 비교적 강함 | 단순 독립 가정 기반이라 복잡한 모델처럼 훈련 데이터에 과하게 맞춰지지 않음 |
| 결과 해석이 직관적 | 확률 기반이라 각 특성이 클래스에 미치는 영향을 직접 해석 가능 |

### 코드 예시

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# 데이터 준비
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# 정규화 (GaussianNB는 필수는 아니지만 비교를 위해 적용)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 가우시안 나이브 베이즈 모델 생성 및 학습
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# 평가
print(f"테스트 정확도: {nb_model.score(X_test, y_test):.4f}")

# 단일 샘플 예측
sample = X_test[0].reshape(1, -1)
prediction = nb_model.predict(sample)
print(f"예측된 클래스: {iris.target_names[prediction[0]]}")
```

> PyTorch 환경에서는 `sklearn.naive_bayes.GaussianNB`를 사용하고 `.numpy()`로 텐서를 변환해서 학습

---

## 알고리즘 비교

| 알고리즘 | 핵심 원리 | 강점 | 약점 |
| --- | --- | --- | --- |
| Random Forest | 여러 의사결정 트리의 앙상블 | 과적합 방지, 특성 중요도 제공 | 학습·예측 시간이 다른 알고리즘보다 길 수 있음 |
| K-NN | K개 이웃의 다수결 | 직관적, 비선형 경계 처리 | 예측 시점에 모든 데이터와 거리 계산 → 대용량에 비효율 |
| SVM | 마진 최대화 초평면 | 고차원에 강함, 이상치에 강건 | 대용량 데이터에서 학습 느림, 커널 선택이 중요 |
| Naive Bayes | 독립 가정 + 베이즈 확률 | 빠르고 단순, 텍스트 분류에 강함 | 특성 간 독립 가정이 현실과 다를 수 있음 |
