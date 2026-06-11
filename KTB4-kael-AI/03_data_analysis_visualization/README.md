# 03_data_analysis_visualization — 데이터 분석 및 시각화

NumPy, Pandas, Matplotlib, Seaborn, SciPy를 활용하여 데이터를 분석하고 시각화하는 과제입니다.

---

## 파일 구성

| 파일 | 내용 | 상태 |
|------|------|------|
| `Numpy.ipynb` | NumPy 배열 생성 및 연산 미니퀘스트 | ✅ 완료 |
| `Pandas.ipynb` | Pandas 데이터 처리 미니퀘스트 (groupby, 필터링 등) | ✅ 완료 |
| `data_visualization.ipynb` | 데이터 시각화 미니퀘스트 + 과제 | ✅ 완료 |
| `netflix_titles.csv` | Netflix 콘텐츠 데이터셋 (Pandas 과제용) |  |

---

## 과제 목록

### 1. NumPy 배열 생성 및 연산 (`Numpy.ipynb`)
- 차원, Shape, 데이터 타입, 인덱싱, 연산, 유니버설 함수 미니퀘스트

### 2. Pandas 데이터 처리 (`Pandas.ipynb`)
- Series, DataFrame, 필터링, 그룹화, 병합, 결측치 처리, 피벗, 중복 제거, 문자열 처리 미니퀘스트
- **2-1.** 데이터를 DataFrame으로 만들고 `groupby` 기능 활용
- **2-2.** 특정 조건을 필터링한 DataFrame 생성 (Netflix 데이터 활용)

### 3. 데이터 시각화 (`data_visualization.ipynb`)
- DataFrame, 데이터 입력 및 처리, 막대 그래프, 히스토그램, 산점도, 박스 플롯, 고급 다중 그래프, 벤 다이어그램 미니퀘스트
- Seaborn 범주형 / 연속형 / 관계 데이터 미니퀘스트
- 시계열 데이터, 리샘플링, 이동평균, 금융 데이터 미니퀘스트
- SciPy 정규 분포, 기술 통계, 가설 검정, 통계적 시각화 미니퀘스트
- **3-1.** yfinance로 AAPL 주가 데이터 입수 후 시계열 시각화

---

## 실행 방법

```bash
# 가상환경 활성화
source .venv/bin/activate

# 필요 패키지 설치
pip install numpy pandas matplotlib seaborn scipy yfinance matplotlib-venn

# Jupyter 실행
jupyter notebook
```

> `data_visualization.ipynb`의 벤 다이어그램 섹션은 `matplotlib-venn` 패키지가 필요합니다.  
> `data_visualization.ipynb`의 3-1 과제(주가 시각화)는 인터넷 연결이 필요합니다.

---

## 폴더 구조

```
03_data_analysis_visualization/
├── README.md
├── Numpy.ipynb                # NumPy 미니퀘스트
├── Pandas.ipynb               # Pandas 미니퀘스트 + 과제 2-1, 2-2
├── data_visualization.ipynb   # 시각화 미니퀘스트 + 과제 3-1
└── netflix_titles.csv         # Pandas 과제용 데이터
```

---

## 회고

<details>
<summary>데이터 분석 (NumPy / Pandas)</summary>

인공지능 연구를 하며 자주 다뤄온 라이브러리라 기본적인 흐름은 익숙했다. 다만 데이터베이스에서 배웠던 `groupby`, `merge` 등 실무적인 집계·조인 연산을 Pandas로 직접 구현해보는 경험은 새로웠다. SQL과 유사한 개념이 Python 코드로 어떻게 표현되는지 비교하며 이해의 폭이 넓어졌다.

</details>

<details>
<summary>데이터 시각화 (Matplotlib / Seaborn / SciPy)</summary>

평소에는 학습 곡선이나 손실 그래프처럼 단순한 선 그래프 위주로만 그려봤는데, 이번 과제를 통해 히스토그램, 박스 플롯, 바이올린 플롯, 벤 다이어그램, 통계적 시각화 등 다양한 그래프를 실제 데이터에 적용해볼 수 있었다. 특히 yfinance로 실시간 주가 데이터를 받아 이동평균과 거래량을 함께 시각화하거나, SciPy로 가설 검정 결과를 그래프로 표현하는 과정에서, 단순히 그래프를 그리는 것을 넘어 데이터에서 의미 있는 정보를 추출하고 시각적으로 전달하는 것이 핵심임을 느꼈다. 앞으로 인공지능 파트에서 모델 학습 데이터를 탐색하거나 결과를 분석할 때 이 경험이 직접적으로 활용될 것 같아 기대된다.

</details>
