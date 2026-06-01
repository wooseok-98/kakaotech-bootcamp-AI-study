# 5-1. SciPy 통계 분석

---

## SciPy (Scientific Python)

과학 계산과 통계 분석을 위한 고급 수학 함수와 알고리즘을 제공하는 파이썬 라이브러리

> NumPy를 기반으로 구축 — 최적화, 보간, 선형 대수, 신호 처리, 통계 분석, 적분 등 고급 기능 제공

### 주요 특징

| 특징 | 설명 |
|------|------|
| NumPy 기반 확장성 | NumPy 배열 기반으로 동작하며 연산을 더 확장 |
| 고급 수학 및 최적화 | 함수 최적화, 최소제곱법, 회귀 분석 등 |
| 신호 처리 | 푸리에 변환(FFT), 필터링, 웨이브렛 변환 |
| 통계 및 확률 분포 | 확률 분포, 가설 검정, 회귀 분석 |
| 선형 대수 | 고유값 분해, LU 분해, 희소 행렬 연산 |
| 보간 및 회귀 | 데이터 사이 값 예측, 다항 회귀 분석 |
| 수치 적분 | `quad`, `dblquad` 등 수치 적분 및 ODE 풀이 |

### 주요 모듈

| 모듈 | 설명 |
|------|------|
| `scipy.optimize` | 함수 최적화, 비선형 방정식 풀이 |
| `scipy.stats` | 확률 분포, 통계 분석, 가설 검정 |
| `scipy.linalg` | 선형 대수 (NumPy보다 확장된 기능) |
| `scipy.fft` | 고속 푸리에 변환(FFT) |
| `scipy.integrate` | 수치 적분, 미분 방정식 풀이 |
| `scipy.interpolate` | 데이터 보간 |
| `scipy.signal` | 신호 처리 (필터링, 컨볼루션) |
| `scipy.sparse` | 희소 행렬 연산 |

### NumPy vs SciPy

| 비교 | NumPy | SciPy |
|------|-------|-------|
| 기능 초점 | 배열 연산, 기본 선형 대수 | 고급 과학 계산, 최적화, 신호 처리, 통계 |
| 최적화 | 없음 | `scipy.optimize` |
| 신호 처리 | 없음 | `scipy.signal` |
| 통계 분석 | 기초 통계 (`mean`, `std` 등) | `scipy.stats` — 확률 분포, 가설 검정 |
| 보간 | 없음 | `scipy.interpolate` |
| 적분 | 없음 | `scipy.integrate` |

### 기본 사용 예시

```python
import numpy as np
from scipy import optimize, stats, signal, integrate, linalg
from scipy.fft import fft

# 1. 최적화 — 함수 최솟값 찾기 (f(x) = x^2 + 5, 최솟값은 x=0)
result = optimize.minimize(lambda x: x**2 + 5, x0=3)
print("최적화 결과:", result.x)  # [-2.83e-08] ≈ 0

# 2. 통계 분석 — 정규 분포에서 난수 생성
samples = stats.norm.rvs(loc=0, scale=1, size=5)

# 3. 신호 처리 — 5Hz 사인파의 FFT
t = np.linspace(0, 1, 500)
sig = np.sin(2 * np.pi * 5 * t)
fft_result = fft(sig)

# 4. 수치 적분 — x^2을 [0, 3] 구간에서 적분 (이론값 = 9)
integral, error = integrate.quad(lambda x: x**2, 0, 3)
print("정적분 결과:", integral)  # 9.000000000000002

# 5. 선형 대수 — 고유값 분해
A = np.array([[4, -2], [1, 1]])
eigenvalues, eigenvectors = linalg.eig(A)
print("고유값:", eigenvalues)  # [3.+0.j 2.+0.j]
```

> **수치 적분** — 컴퓨터가 적분 구간을 잘게 나눠 면적을 근사적으로 계산하는 방법. 부동소수점 오차로 인해 이론값과 미세하게 다를 수 있음

---

## 정규 분포 (Normal Distribution)

데이터가 평균을 중심으로 좌우 대칭을 이루며 종형 곡선을 따르는 확률 분포

```
f(x) = (1 / (σ × √(2π))) × exp(-(x-μ)² / (2σ²))
```

| 파라미터 | 설명 |
|---------|------|
| `μ` (평균) | 분포의 중심 위치 결정 |
| `σ` (표준편차) | 작으면 좁고 뾰족, 크면 넓고 완만 |

### 주요 특징

| 특징 | 설명 |
|------|------|
| 좌우 대칭 | 평균을 기준으로 대칭 |
| **68-95-99.7 법칙** | μ±1σ 내 68%, μ±2σ 내 95%, μ±3σ 내 99.7% |
| 통계 기법의 기본 가정 | t-검정, 회귀 분석, PCA 등에서 정규 분포 가정 |

### Z-점수 변환 (표준화)

평균 0, 표준편차 1인 표준 정규 분포로 변환 — 서로 다른 분포 간 비교 가능

```
Z = (X - μ) / σ
```

| Z값 | 의미 |
|-----|------|
| 0 | 평균과 동일 |
| +1 / -1 | 평균보다 1σ 높음/낮음 |
| +2 / -2 | 상위/하위 약 2.5% |

### 사용 방법

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=1000)  # 평균 50, 표준편차 10

# PDF 시각화
plt.figure(figsize=(8, 5))
sns.histplot(data, bins=30, kde=True, color="skyblue", alpha=0.7)
plt.title("Normal Distribution (μ=50, σ=10)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# 특정 값의 확률 밀도 (PDF)
pdf_value = stats.norm.pdf(55, loc=50, scale=10)   # → 0.0352

# 누적 확률 (CDF) — x=55 이하일 확률
cdf_value = stats.norm.cdf(55, loc=50, scale=10)   # → 0.6915 (약 69%)

# 분위수 (PPF) — 상위 10%의 경계값
quantile_90 = stats.norm.ppf(0.90, loc=50, scale=10)  # → 62.82

# 이상값 탐지 (평균 ± 2σ 기준)
lower, upper = 50 - 2 * 10, 50 + 2 * 10
outliers = data[(data < lower) | (data > upper)]
print(f"이상값 개수: {len(outliers)}")  # 약 50개 (전체의 5%)
```

> **확률 밀도 함수(PDF)** — y값이 곧 확률(%)은 아님. 구간의 곡선 아래 면적이 실제 확률. y값은 해당 구간에 데이터가 얼마나 밀집해 있는지를 나타내는 밀도

---

## 기술 통계 (Descriptive Statistics)

데이터를 요약·정리하여 중앙 경향성, 산포도, 분포 형태 등의 지표를 계산하고 시각적으로 표현하는 분석 기법

### 핵심 지표

**중앙 경향성 (Central Tendency)**

| 지표 | 설명 |
|------|------|
| **평균(Mean)** | 값의 합 ÷ 개수, 이상값의 영향을 받음 |
| **중앙값(Median)** | 정렬 후 가운데 값, 이상값 영향 없음 |
| **최빈값(Mode)** | 가장 자주 등장하는 값 |

**산포도 (Dispersion)**

| 지표 | 설명 |
|------|------|
| **범위(Range)** | 최댓값 - 최솟값 |
| **분산(Variance)** | 평균으로부터의 편차 제곱 평균 |
| **표준편차(Std)** | 분산의 제곱근, 변동성을 직관적으로 파악 |
| **IQR** | Q3 - Q1, 중간 영역의 퍼짐 정도 |

**분포 특성 (Distribution)**

| 지표 | 설명 |
|------|------|
| **왜도(Skewness)** | 0보다 크면 오른쪽 치우침(양의 왜도), 작으면 왼쪽 |
| **첨도(Kurtosis)** | 3보다 크면 뾰족(급첨), 작으면 평평(완만) |

### 사용 방법

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=100)
df = pd.DataFrame(data, columns=["value"])

# 1. 한 번에 기본 통계 확인
print(df.describe())
# count, mean, std, min, 25%, 50%, 75%, max 출력

# 2. 개별 지표 계산
mean_val   = np.mean(df["value"])
median_val = np.median(df["value"])
mode_val   = stats.mode(df["value"], keepdims=True).mode[0]
std_val    = np.std(df["value"], ddof=1)    # ddof=1: 표본분산(불편분산)
var_val    = np.var(df["value"], ddof=1)
range_val  = np.ptp(df["value"])

q1 = np.percentile(df["value"], 25)
q3 = np.percentile(df["value"], 75)
iqr = q3 - q1

skewness  = stats.skew(df["value"])
kurtosis  = stats.kurtosis(df["value"])

# 3. 이상값 탐지 (IQR 기준)
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = df[(df["value"] < lower) | (df["value"] > upper)]
print(f"이상값 개수: {len(outliers)}")

# 4. 히스토그램 시각화
plt.figure(figsize=(8, 5))
sns.histplot(df["value"], bins=15, kde=True, color="skyblue", alpha=0.6)
plt.title("Data Distribution Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

> **ddof=1** — 표본 데이터로 모집단 분산을 추정할 때 n 대신 n-1로 나눠 과소평가를 보정 (불편분산). ddof=0이면 모분산

---

## 가설 검정 (Hypothesis Testing)

표본을 기반으로 통계적 가설의 참과 거짓을 검정하여 결론을 도출하는 과정

> 시각적으로 보이는 차이가 우연인지, 통계적으로 유의미한 차이인지 판단

### 핵심 개념

| 개념 | 설명 |
|------|------|
| **귀무가설(H₀)** | "차이가 없다"는 기본 가정 — 반증 대상 |
| **대립가설(H₁)** | "차이가 있다"는 주장 — 증명하려는 가설 |
| **검정 통계량** | 표본에서 계산한 값 (t값, F값 등) |
| **p값** | 귀무가설이 참일 때 현재만큼 극단적인 결과가 나올 확률 |
| **유의수준(α)** | 귀무가설 기각 기준 — 보통 0.05 사용 |
| **기각역** | 검정 통계량이 이 영역에 속하면 H₀ 기각 |
| **1종 오류** | 실제 H₀이 참인데 기각 ("효과 없는데 있다고 착각") |
| **2종 오류** | 실제 H₁이 참인데 기각 안 함 ("효과 있는데 없다고 판단") |

### 검정 절차

| 순서 | 단계 | 예시 |
|------|------|------|
| 1 | 가설 설정 | H₀: "두 그룹 평균 차이 없음", H₁: "차이 있음" |
| 2 | 유의수준 결정 | α = 0.05 |
| 3 | 검정 통계량 계산 | t-검정, 카이제곱 검정 등 |
| 4 | p값 계산 및 해석 | p < 0.05 → H₀ 기각 |
| 5 | 결론 도출 | "유의미한 차이 있음" 또는 "차이 없음" |

### 검정 유형

| 검정 종류 | 설명 |
|---------|------|
| **단일 표본 t-검정** | 한 그룹 평균이 특정 값과 다른지 — "평균 키가 170cm인가?" |
| **독립 표본 t-검정** | 두 그룹 평균 차이 — "남성과 여성의 평균 체온이 같은가?" |
| **대응 표본 t-검정** | 동일 그룹 전/후 비교 — "운동 전후 체중 변화가 있는가?" |
| **카이제곱 검정** | 범주형 변수 간 독립성 — "흡연과 폐암 발병률 연관 있는가?" |
| **ANOVA** | 3개 이상 그룹 평균 비교 — "3개 학급 시험 성적 차이 있는가?" |
| **회귀 분석** | 변수 간 영향 검정 — "광고비가 매출에 영향을 주는가?" |

### 사용 방법

| 검정 방법 | 기본 문법 | 설명 |
|---------|---------|------|
| 단일 표본 t-검정 | `stats.ttest_1samp(data, popmean)` | 한 그룹 평균이 특정 값과 다른지 |
| 독립 표본 t-검정 | `stats.ttest_ind(sample1, sample2)` | 두 독립 그룹의 평균 차이 |
| 대응 표본 t-검정 | `stats.ttest_rel(before, after)` | 동일 그룹의 전/후 비교 |
| 카이제곱 검정 | `stats.chi2_contingency(observed)` | 범주형 변수 간 독립성 |
| ANOVA | `stats.f_oneway(g1, g2, g3)` | 3개 이상 그룹 평균 비교 |
| 회귀 분석 | `stats.linregress(x, y)` | 변수 간 영향 검정 |

```python
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
group_A = np.random.normal(loc=55, scale=10, size=100)
group_B = np.random.normal(loc=50, scale=10, size=100)

# 독립 표본 t-검정
t_stat, p_value = stats.ttest_ind(group_A, group_B, equal_var=False)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
# t-statistic: 2.8388, p-value: 0.0050 → p < 0.05이므로 H₀ 기각 (차이 유의미)

# 결과 시각화
df_ab = pd.DataFrame({
    "value": np.concatenate([group_A, group_B]),
    "group": ["A"] * 100 + ["B"] * 100
})
plt.figure(figsize=(6, 5))
sns.boxplot(x="group", y="value", hue="group", data=df_ab, palette=["lightblue", "lightcoral"], legend=False)
plt.title("A/B Test - Purchase Amount Comparison")
plt.xlabel("Group")
plt.ylabel("Purchase Amount")
plt.show()
```

```python
# 카이제곱 검정 — 광고 A/B 그룹 구매 여부
observed = pd.DataFrame({
    "Ad_A": [40, 60],
    "Ad_B": [30, 70]
}, index=["Purchase", "No Purchase"])

chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
print(f"Chi-Square: {chi2_stat:.4f}, p-value: {p_value:.4f}")
# Chi-Square: 1.7802, p-value: 0.1821 → p > 0.05이므로 H₀ 유지 (차이 유의미하지 않음)

# 결과 시각화
plt.figure(figsize=(6, 5))
sns.barplot(x=observed.columns, y=observed.loc["Purchase"], hue=observed.columns,
            palette=["lightblue", "lightcoral"])
plt.title("Ad A vs. Ad B - Purchase Rate")
plt.xlabel("Advertisement Group")
plt.ylabel("Number of Purchases")
plt.show()
```

---

## 통계적 시각화 (Statistical Visualization)

데이터의 분포, 관계, 추세 등을 효과적으로 분석하기 위해 통계적 기법을 활용하여 그래프나 차트로 표현하는 과정

### 분석 유형

| 구분 | 설명 | 대표 시각화 |
|------|------|-----------|
| **단변량(Univariate)** | 하나의 변수 분포 및 특성 분석 | 히스토그램, KDE, 박스플롯 |
| **이변량(Bivariate)** | 두 변수 간의 관계 분석 | 산점도, 상관 행렬, 회귀선 |
| **다변량(Multivariate)** | 세 개 이상 변수 간 관계 분석 | 페어플롯, 히트맵, 3D 산점도 |

### Matplotlib 기본 시각화

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=1000)
df = pd.DataFrame(data, columns=["value"])

# 히스토그램
plt.figure(figsize=(8, 5))
plt.hist(df["value"], bins=30, color="skyblue", edgecolor="black", alpha=0.7)
plt.title("Histogram of Data Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# 박스플롯
plt.figure(figsize=(6, 4))
plt.boxplot(df["value"], vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("Boxplot of Data Distribution")
plt.xlabel("Value")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

# 산점도
plt.figure(figsize=(8, 5))
plt.scatter(range(len(df)), df["value"], alpha=0.5, color="blue", edgecolors="black")
plt.title("Scatter Plot of Data Distribution")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(linestyle="--", alpha=0.7)
plt.show()
```

### Seaborn 확장 시각화

```python
import seaborn as sns

# 히스토그램 + KDE
plt.figure(figsize=(8, 5))
sns.histplot(df["value"], bins=30, kde=True, color="skyblue", alpha=0.7)
plt.title("Histogram with KDE")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

# 박스플롯
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["value"], color="lightblue")
plt.title("Boxplot using Seaborn")
plt.xlabel("Value")
plt.show()

# 바이올린 플롯 — 박스플롯 + KDE 결합, 분포 형태를 더 상세하게 표현
plt.figure(figsize=(6, 4))
sns.violinplot(x=df["value"], color="lightblue")
plt.title("Violin Plot of Data")
plt.xlabel("Value")
plt.show()
```

### SciPy + Seaborn 통합 활용

가설 검정 후 결과를 시각화하면 통계적 차이를 직관적으로 확인 가능

```python
# 독립 표본 t-검정 + 박스플롯
np.random.seed(42)
group_A = np.random.normal(loc=55, scale=10, size=100)
group_B = np.random.normal(loc=50, scale=10, size=100)

t_stat, p_value = stats.ttest_ind(group_A, group_B, equal_var=False)
# t-statistic: 2.8388, p-value: 0.0050 → 두 그룹 차이 유의미

df_ab = pd.DataFrame({
    "value": np.concatenate([group_A, group_B]),
    "group": ["A"] * 100 + ["B"] * 100
})
plt.figure(figsize=(6, 5))
sns.boxplot(x="group", y="value", hue="group", data=df_ab,
            palette=["lightblue", "lightcoral"], legend=False)
plt.title("A/B Test - Purchase Amount Comparison")
plt.xlabel("Group")
plt.ylabel("Purchase Amount")
plt.show()

# 카이제곱 검정 + 막대 그래프
observed = pd.DataFrame({"Ad_A": [40, 60], "Ad_B": [30, 70]},
                        index=["Purchase", "No Purchase"])
chi2_stat, p_value, _, _ = stats.chi2_contingency(observed)
# Chi-Square: 1.7802, p-value: 0.1821 → 구매율 차이 유의미하지 않음

plt.figure(figsize=(6, 5))
sns.barplot(x=observed.columns, y=observed.loc["Purchase"], hue=observed.columns,
            palette=["lightblue", "lightcoral"])
plt.title("Ad A vs. Ad B - Purchase Rate")
plt.xlabel("Advertisement Group")
plt.ylabel("Number of Purchases")
plt.show()
```
