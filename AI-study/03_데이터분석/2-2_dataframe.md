# 2-2. 데이터프레임 (DataFrame)

**행과 열로 구성된 2차원 테이블 형태의 데이터 구조**

각 열은 Series로 구성되며, 엑셀 스프레드시트나 데이터베이스 테이블과 유사한 형태

---

## 구성 요소

| 요소 | 설명 |
| --- | --- |
| **values** | 실제 데이터. 각 열은 개별 데이터 타입을 가질 수 있음 |
| **column** | 열 이름(레이블). 각 열은 Series 객체 |
| **row** | 행. 서로 다른 속성의 데이터를 포함하며 인덱스로 접근 |
| **index** | 각 행을 식별하는 레이블 (기본: 0부터 시작 정수) |

---

## 1. 생성

```python
import pandas as pd
import numpy as np

# 딕셔너리 — 키가 열 이름
data = {'이름': ['홍길동', '김철수', '박영희'],
        '나이': [25, 30, 28],
        '성별': ['남', '남', '여']}
df = pd.DataFrame(data)

# 리스트 — columns로 열 이름 지정
df2 = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])

# NumPy 배열
df3 = pd.DataFrame(np.array([[10, 20], [30, 40]]), columns=['X', 'Y'])
```

---

## 2. 기본 속성

```python
print(df.head())        # 처음 5개 행
print(df.tail())        # 마지막 5개 행
print(df.shape)         # (3, 3) — 행 × 열
print(df.columns)       # 열 이름 목록
print(df.index)         # 행 인덱스 범위
print(df.info())        # 타입, 결측치 여부 요약
print(df.describe())    # 수치형 데이터 통계 요약 (평균, 표준편차, 최댓값 등)
```

---

## 3. 데이터 접근

```python
# 열 선택
print(df['이름'])
print(df[['이름', '나이']])   # 여러 열

# 행 선택 — iloc: 정수 기반
print(df.iloc[0])

# 행 선택 — loc: 레이블 기반
df = df.set_index('이름')
print(df.loc['홍길동'])

# 특정 셀
print(df.loc['홍길동', '나이'])
```

---

## 4. 데이터 수정 및 연산

```python
# 값 수정
df.loc['홍길동', '나이'] = 26

# 열 추가
df['국적'] = '한국'

# 열 삭제
df.drop('국적', axis=1, inplace=True)

# 새 열 생성 (기존 열 연산)
df['총점'] = df['국어'] + df['영어'] + df['수학']

# 조건으로 행 추출
high = df[df['국어'] >= 90]
```

---

## 5. 정렬

```python
df.sort_values(by='나이')                      # 오름차순
df.sort_values(by='나이', ascending=False)     # 내림차순
```

---

## 6. 그룹화

```python
df.groupby('성별')['국어'].mean()
# 성별별 국어 평균
```

---

## 7. 결측치 처리

```python
df.isnull().sum()           # 결측치 확인
df.dropna(inplace=True)     # 결측치 행 제거
df.fillna(0, inplace=True)  # 결측치를 0으로 채우기
```

---

## 8. CSV 저장 및 로딩

```python
df.to_csv('data.csv', index=False)
df_loaded = pd.read_csv('data.csv')
```
