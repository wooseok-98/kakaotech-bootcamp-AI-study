# 2-1. Matplotlib

파이썬에서 다양한 형태의 차트와 그래프를 생성할 수 있도록 지원하는 기본적인 데이터 시각화 라이브러리

> MATLAB(Matrix Laboratory)의 plotting 기능을 파이썬으로 구현한 라이브러리

---

## 주요 특징

| 특징 | 설명 |
|------|------|
| 광범위한 그래프 지원 | 라인, 막대, 히스토그램, 산점도, 박스플롯 등 |
| 커스터마이징 | 색상, 선 스타일, 마커, 폰트, 축 범위 등 세밀한 조정 가능 |
| 객체 지향 방식 | Figure(전체 도화지) + Axes(개별 그래프) 구조 |
| 다양한 저장 형식 | PNG, PDF, SVG 등 |
| 호환성 | Pandas, Seaborn과 연동 가능 |

---

## 기본 사용법

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 25, 30]

plt.plot(x, y)
plt.title("Basic Line Graph")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

> `plt.show()`는 스크립트 환경에서 필수. Jupyter/Colab에서는 없어도 자동 출력됨

---

## Pandas 내장 시각화

Pandas는 Matplotlib 기반의 `DataFrame.plot()` 메서드를 제공 — 코드 한 줄로 간단하게 시각화 가능

```python
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
    "Sales": [100, 120, 90, 150, 200],
    "Cost":  [80,  95,  70, 110, 140]
}
df = pd.DataFrame(data)

df.plot(x="Month", y=["Sales", "Cost"], kind="line", marker="o", figsize=(8, 5))
plt.title("Monthly Sales and Cost")
plt.show()
```

| `kind` 값 | 그래프 종류 |
|-----------|------------|
| `"line"` | 선 그래프 |
| `"bar"` | 막대 그래프 |
| `"hist"` | 히스토그램 |
| `"box"` | 박스 플롯 |
| `.scatter(x, y)` | 산점도 |

**Pandas vs Matplotlib 비교**

| 항목 | Pandas 내장 | Matplotlib |
|------|------------|------------|
| 사용법 | `df.plot(kind="")` 한 줄 | `plt.figure()`, `plt.plot()` 등 여러 단계 |
| 커스터마이징 | 기본 수준 | 세부 조정 가능 |
| 목적 | 빠른 데이터 탐색 | 고급 시각화, 발표용 차트 |

---

## 막대 그래프 (Bar Chart)

범주형 데이터의 크기 비교를 위해 각 범주를 막대의 길이로 표현하는 시각화 방법

- 언제 사용: 제품별 매출, 지역별 인구, 설문 결과, 클래스 불균형 확인 등

### 기본 문법

```python
plt.bar(x, height, color='색상', label='라벨')   # 세로 막대
plt.barh(x, height, color='색상', label='라벨')  # 가로 막대
```

### 주요 유형

```python
import matplotlib.pyplot as plt
import numpy as np

categories = ['A', 'B', 'C', 'D', 'E']
values1 = [10, 20, 15, 25, 30]
values2 = [5, 10, 10, 15, 20]
x = np.arange(len(categories))

# 1. 기본 막대 그래프
plt.bar(categories, values1, color='skyblue')

# 2. 가로 막대 그래프
plt.barh(categories, values1, color='lightcoral')

# 3. 누적형 (Stacked)
plt.bar(x, values1, color='dodgerblue', label='2023')
plt.bar(x, values2, color='orange', bottom=values1, label='2024')  # bottom으로 누적
plt.xticks(x, categories)
plt.legend()

# 4. 그룹형 (Grouped)
bar_width = 0.4
plt.bar(x - bar_width/2, values1, width=bar_width, color='green', label='2023')
plt.bar(x + bar_width/2, values2, width=bar_width, color='purple', label='2024')
plt.xticks(x, categories)
plt.legend()

plt.show()
```

> **범례(legend)**: 색상/선이 무엇을 의미하는지 설명하는 안내표. `plt.legend()`로 추가

---

## 히스토그램 (Histogram)

연속형 데이터의 분포를 나타내기 위해 데이터를 구간(bin)으로 나누고, 각 구간의 빈도수를 막대로 표현하는 시각화 방법

- 막대 그래프와의 차이: 막대 그래프는 범주 비교, 히스토그램은 **연속형 데이터의 분포 확인**
- **EDA(탐색적 데이터 분석)** 에서 필수적으로 사용

### 구성 요소

| 용어 | 설명 |
|------|------|
| bin | 데이터를 나누는 구간 단위. 너무 많으면 노이즈, 너무 적으면 패턴 손실 |
| 빈도수(Frequency) | 각 구간에 속하는 데이터 개수 |
| 봉우리(Peak) | 데이터가 특정 값 주변에 많이 분포하는 지점 |

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(1000)   # 표준 정규분포 (평균 0, 표준편차 1) → x축이 -3~+3인 이유

# 기본 히스토그램
plt.hist(data, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Basic Histogram")

# 구간 수 비교
plt.hist(data, bins=10, color='lightgreen', alpha=0.7, label='bins=10')
plt.hist(data, bins=30, color='salmon',     alpha=0.7, label='bins=30')
plt.legend()

# 두 그룹 비교
data2 = np.random.randn(1000) + 2
plt.hist(data,  bins=20, alpha=0.5, label='Group 1')
plt.hist(data2, bins=20, alpha=0.5, label='Group 2')
plt.legend()

plt.show()
```

---

## 산점도 (Scatter Plot)

변수 간의 관계를 나타내기 위해 각 데이터 점을 좌표 평면에 점으로 표현하는 시각화 방법

- 언제 사용: 변수 간 상관관계, 이상치 탐지, 군집 구조 확인, 모델 예측값 vs 실제값 비교

### 상관관계 유형

- 양의 상관관계: 점들이 우상향 → 공부 시간↑ 시험 점수↑
- 음의 상관관계: 점들이 우하향
- 무상관: 점들이 흩어져 있음

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
x = np.random.randn(50)
y = np.random.randn(50)

# 기본 산점도
plt.scatter(x, y, color='blue', marker='o')

# 커스터마이징
plt.scatter(x, y, color='green', marker='^', alpha=0.7, s=100)
# marker 종류: 'o'원 '^'삼각형 's'사각형 '*'별 'x'X표 'D'다이아몬드

# 카테고리별 색상 구분
categories = np.random.choice(['A', 'B', 'C'], size=50)
colors = {'A': 'red', 'B': 'blue', 'C': 'green'}
for cat in np.unique(categories):
    idx = categories == cat
    plt.scatter(x[idx], y[idx], color=colors[cat], label=f'Category {cat}', alpha=0.7)
plt.legend()

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot")
plt.show()
```

---

## 박스 플롯 (Box Plot)

데이터의 분포, 중앙값, 사분위수, 이상치를 시각적으로 나타내는 시각화 방법

- 언제 사용: 이상치 탐지, 그룹 간 분포 비교, 데이터 스케일 확인

### 구성 요소

| 요소 | 설명 |
|------|------|
| 박스(Box) | Q1~Q3 범위 — 중앙 50% 데이터 |
| 중앙값(Q2) | 박스 내 굵은 선 |
| Q1 (1사분위) | 하위 25% 지점 |
| Q3 (3사분위) | 상위 25% 지점 |
| IQR | Q3 - Q1 (박스의 폭) |
| 수염(Whiskers) | Q1 - 1.5×IQR ~ Q3 + 1.5×IQR 범위 |
| 이상치(Outlier) | 수염 범위를 벗어난 값 — 점(○)으로 표시 |

> IQR의 1.5배 기준은 통계학자 Tukey가 제안한 경험적 규칙 (고정값 아님, 분석 목적에 따라 조정 가능)

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
data1 = np.random.randn(100)
data2 = np.random.randn(100) + 2
data3 = np.random.randn(100) - 2

# 기본 박스 플롯
plt.boxplot(data1)

# 다중 비교
plt.boxplot([data1, data2, data3], tick_labels=['Group 1', 'Group 2', 'Group 3'])

# 스타일 커스터마이징
plt.boxplot(data1,
            patch_artist=True,                                       # 박스 내부 색 채우기 (True=면, False=선)
            boxprops=dict(facecolor="lightblue", color="blue"),      # 박스 색상
            whiskerprops=dict(color="red", linewidth=2),             # 수염 스타일
            medianprops=dict(color="black", linewidth=2))            # 중앙값 선

# 이상치 강조
plt.boxplot(data1, flierprops=dict(marker='o', markerfacecolor='red', markersize=10))

# 수평 박스 플롯
plt.boxplot(data1, vert=False, patch_artist=True, boxprops=dict(facecolor="lightgray"))

plt.title("Box Plot")
plt.ylabel("Values")
plt.show()
```

---

## 고급 다중 그래프 (Advanced Multiple Graphs)

하나의 Figure 내에서 여러 개의 그래프를 배치하여 다양한 데이터 관계를 동시에 시각화하는 기법

- 언제 사용: 여러 변수 비교, AI 모델 학습 지표(loss·accuracy) 동시 시각화, 실험 결과 비교

### 주요 패턴

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1, y2 = np.sin(x), np.cos(x)

# 1. 세로 배치 (2행 1열)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
axes[0].plot(x, y1, color='b', label='sin(x)')
axes[0].set_title("Sine")
axes[0].legend()
axes[1].plot(x, y2, color='r', label='cos(x)')
axes[1].set_title("Cosine")
axes[1].legend()
plt.tight_layout()

# 2. 가로 배치 (1행 2열)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
axes[0].plot(x, y1, color='g')
axes[1].plot(x, y2, color='m')
plt.tight_layout()

# 3. 다양한 그래프 조합 (2행 2열)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
data = np.random.randn(100)
axes[0, 0].plot(x, y1, color='b');          axes[0, 0].set_title("Line Plot")
axes[0, 1].scatter(x, y1, color='r');       axes[0, 1].set_title("Scatter Plot")
axes[1, 0].bar(np.arange(5), [3,1,4,1,5]); axes[1, 0].set_title("Bar Chart")
axes[1, 1].hist(data, bins=20, color='purple'); axes[1, 1].set_title("Histogram")
plt.tight_layout()

# 4. 공유 축 (X축 공유)
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
axes[0].plot(x, y1, color='b')
axes[1].plot(x, y2, color='r')
plt.tight_layout()

plt.show()
```

### GridSpec — 불규칙한 레이아웃

```python
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 2, figure=fig)

ax1 = fig.add_subplot(gs[0, :])    # 첫 번째 행 전체
ax2 = fig.add_subplot(gs[1, 0])    # 두 번째 행 왼쪽
ax3 = fig.add_subplot(gs[1, 1])    # 두 번째 행 오른쪽
ax4 = fig.add_subplot(gs[2, :])    # 세 번째 행 전체

ax1.plot(x, np.sin(x),   color='b', label="sin(x)");  ax1.legend()
ax2.plot(x, np.cos(x),   color='r', label="cos(x)");  ax2.legend()
ax3.plot(x, np.exp(-x/5), color='g', label="exp");    ax3.legend()
ax4.plot(x, np.log(x+1), color='purple', label="log"); ax4.legend()

plt.tight_layout()
plt.show()
```

---

## 벤 다이어그램 (Venn Diagram)

여러 집합 간의 관계와 교집합을 시각적으로 표현하는 다이어그램

- 1880년 영국 논리학자 존 벤(John Venn)이 처음 소개
- 언제 사용: 고객 세그먼트 중복 확인, Train/Test 데이터 중복 검증, 모델 예측 결과 비교

```python
# 설치 필요
# pip install matplotlib-venn
from matplotlib_venn import venn2, venn3
import matplotlib.pyplot as plt

# 1. 기본 벤 다이어그램 (집합 직접 전달)
set_A = {"사과", "바나나", "체리", "망고"}
set_B = {"바나나", "망고", "포도", "수박"}
venn2([set_A, set_B], set_labels=("Set A", "Set B"))
plt.title("Basic Venn Diagram")
plt.show()

# 2. 3개 집합
set_C = {"망고", "수박", "딸기", "오렌지"}
venn3([set_A, set_B, set_C], set_labels=("Set A", "Set B", "Set C"))
plt.title("Venn Diagram with Three Sets")
plt.show()

# 3. 값 직접 설정
# subsets=(A만, B만, A∩B)
venn2(subsets=(3, 4, 2), set_labels=("Set A", "Set B"))

# 4. 색상 커스터마이징
venn2(subsets=(3, 4, 2), set_labels=("Set A", "Set B"),
      set_colors=("skyblue", "lightcoral"))

# 5. 특정 영역 강조
# "10"=A만, "01"=B만, "11"=교집합
diagram = venn2(subsets=(3, 4, 2), set_labels=("Set A", "Set B"))
diagram.get_label_by_id("11").set_color("red")
diagram.get_label_by_id("11").set_fontsize(14)

plt.show()
```
