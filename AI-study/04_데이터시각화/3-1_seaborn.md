# 3-1. Seaborn

통계적 데이터 시각화를 쉽게 구현할 수 있도록 고급 스타일과 기능을 제공하는 파이썬 데이터 시각화 라이브러리

> Matplotlib 기반으로 동작 — Pandas DataFrame과 직접 연동 가능

---

## 주요 특징

| 특징 | 설명 |
|------|------|
| 통계적 시각화 | 회귀선, 밀도 추정(KDE), 신뢰구간 등 통계 요소 내장 |
| 세련된 기본 스타일 | 색상 팔레트, 축 스타일, 격자 자동 조정 |
| 간결한 코드 | Matplotlib보다 짧은 코드로 고급 시각화 구현 |
| DataFrame 연동 | `data=df, x='컬럼명'` 형태로 바로 사용 가능 |

> **Matplotlib vs Seaborn 학습 방식 차이**  
> Matplotlib → 그래프 단위 학습 (막대, 히스토그램 등)  
> Seaborn → **데이터 유형 단위** 학습 (범주형, 연속형, 관계형) — 데이터 특성에 맞는 시각화를 자동으로 최적화하기 때문

---

## 기본 사용법

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({
    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "y": [2, 4, 1, 5, 7, 8, 5, 9, 10, 6]
})

sns.scatterplot(x="x", y="y", data=df)
plt.title("Basic Scatter Plot with Seaborn")
plt.show()
```

---

## 범주형 데이터 (Categorical Data)

정해진 그룹이나 레이블을 가지는 데이터 — 예) 성별, 등급, 제품 종류

- **명목형(Nominal)**: 순서 없는 범주 (성별, 혈액형)
- **순서형(Ordinal)**: 순서 있는 범주 (만족도, 교육 수준)

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame({
    "Category": ["A", "A", "B", "B", "C", "C", "C", "A", "B", "C"],
    "Value":    [10,  15,  7,  12,  22,  18,  25,  11,   9,  30]
})

# 1. 막대 그래프 — 범주별 평균 + 신뢰구간(오차 막대 | ) 자동 표시
sns.barplot(x="Category", y="Value", data=data)
# ci=None 으로 오차 막대 제거 가능

# 2. 박스 플롯 — 범주별 분포
sns.boxplot(x="Category", y="Value", data=data)

# 3. 바이올린 플롯 — 박스플롯 + KDE(밀도 분포) 동시 표현
sns.violinplot(x="Category", y="Value", data=data)
# 박스플롯보다 분포의 형태(비대칭성, 다봉형)까지 파악 가능

# 4. 스트립 플롯 — 개별 데이터 포인트를 점으로 표시
sns.stripplot(x="Category", y="Value", data=data, jitter=True)
# jitter=True: 점이 겹치지 않도록 위치를 약간 흩어줌

plt.title("Categorical Plot")
plt.show()
```

### 그래프 선택 기준

| 그래프 | 장점 | 추천 상황 |
|--------|------|-----------|
| 막대 그래프 | 범주별 평균 비교 명확 | 그룹 간 평균 차이 파악 |
| 박스 플롯 | 요약 통계(중앙값·IQR·이상치) 한눈에 | 분포 요약 및 이상치 확인 |
| 바이올린 플롯 | 분포 형태까지 표현 | 분포 모양(비대칭, 다봉형) 분석 |
| 스트립 플롯 | 개별 값 직접 확인 | 표본 수 적을 때, 실제 데이터 확인 필요 시 |

---

## 연속형 데이터 (Continuous Data)

특정 구간 내에서 무한한 값을 가질 수 있는 데이터 — 예) 키, 온도, 연봉

### 빈도 vs 확률 밀도

| 구분 | 설명 | 시각화 |
|------|------|--------|
| 빈도(Frequency) | 구간(bin)별 데이터 개수 | histplot |
| 확률 밀도(KDE) | 부드러운 분포 곡선 — 히스토그램보다 연속적 | kdeplot |

```python
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)   # 시드 42: The Hitchhiker's Guide to the Galaxy에서 유래한 관습
data = np.random.randn(1000)

x = np.linspace(0, 10, 100)
np.random.seed(0)
x2 = np.random.rand(100) * 10
y2 = x2 + np.random.randn(100)

# 1. 히스토그램
sns.histplot(data, bins=30, color='steelblue')

# 2. 히스토그램 + KDE 동시 표현
sns.histplot(data, bins=30, kde=True, color='darkorange')
# kde=True: 확률 밀도 곡선 추가

# 3. 선 그래프
sns.lineplot(x=x, y=np.sin(x), color='royalblue')

# 4. 산점도
sns.scatterplot(x=x2, y=y2, color='crimson')

# 5. 회귀선 포함 산점도
sns.regplot(x=x2, y=y2, color='green')
# 회귀선: 두 변수 간 관계를 수학적 모델(직선)로 표현한 경향선

plt.show()
```

> **KDE(커널 밀도 추정)**: 각 데이터 포인트 주변에 가우시안 커널 함수를 씌워 부드러운 분포 곡선을 추정하는 방법

---

## 관계 데이터 (Relational Data)

두 개 이상의 변수 간의 상관관계나 패턴을 분석하는 데이터

- 양의 상관관계: 한 변수 증가 → 다른 변수 증가
- 음의 상관관계: 한 변수 증가 → 다른 변수 감소
- 무상관: 관계 없음

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")   # Seaborn 내장 식당 팁 데이터셋

# 1. 산점도 — 기본 관계 탐색
sns.scatterplot(x="total_bill", y="tip", data=tips, color="blue")

# 2. 회귀선 포함 산점도
sns.regplot(x="total_bill", y="tip", data=tips,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'})

# 3. 페어 플롯 — 여러 변수 간 관계를 한 번에 비교
sns.pairplot(tips, hue="sex")   # hue로 범주별 색상 구분
# 대각선: 각 변수의 분포 / 나머지: 변수 쌍의 산점도

plt.show()
```

> **페어 플롯(Pair Plot)**: 데이터셋의 모든 수치형 변수 쌍을 한 Figure에 한꺼번에 시각화 — EDA 초기에 변수 간 상관관계 탐색에 유용
