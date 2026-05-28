# 3-4. 결측치 처리 & 중복 제거

---

## 결측치 처리 (Missing Data)

**데이터프레임이나 시리즈에서 누락된 값을 탐지하고 제거하거나 대체하는 작업**

Pandas에서 결측치는 `NaN` 또는 `None`으로 표시

| 단계 | 메서드 |
| --- | --- |
| 탐지 | `isnull()`, `notnull()`, `info()` |
| 제거 | `dropna()` |
| 대체 | `fillna()`, `ffill()`, `bfill()` |

### 1. 결측치 확인

```python
import pandas as pd
import numpy as np

data = {'이름': ['홍길동', '김철수', np.nan, '이영희'],
        '나이': [25, np.nan, 30, 28],
        '성별': ['남', '남', '여', np.nan]}
df = pd.DataFrame(data)

print(df.isnull())          # 요소별 True/False
print(df.isnull().sum())    # 열별 결측치 개수
df.info()
```

> 정수 열에 NaN이 섞이면 float64로 자동 변환됨

### 2. 결측치 제거 (`dropna`)

```python
df.dropna()                   # 결측치 있는 행 전체 삭제
df.dropna(subset=['이름'])     # 특정 열 기준으로만 삭제
df.dropna(axis=1)             # 결측치 있는 열 삭제
```

### 3. 결측치 대체 (`fillna`)

```python
# 특정 값으로 대체
df.fillna({'이름': '없음', '나이': 0, '성별': '없음'})

# 평균값으로 대체
df['나이'] = df['나이'].fillna(df['나이'].mean())

# 이전 값으로 채우기 (forward fill)
df.ffill()

# 이후 값으로 채우기 (backward fill)
df.bfill()
```

### 4. 결측치 기반 필터링

```python
df[df['이름'].notnull()]    # 결측치 없는 행만 선택
df[df['성별'].isnull()]     # 결측치 있는 행만 선택
```

---

## 중복 제거 (Duplicates Removal)

**데이터프레임이나 시리즈에서 동일한 값을 갖는 중복된 행을 식별하고 제거하는 작업**

| 메서드 | 설명 |
| --- | --- |
| `duplicated()` | 중복 여부 확인 — True/False 반환 |
| `drop_duplicates()` | 중복 행 제거 |

### 1. 중복 확인 (`duplicated`)

```python
data = {
    '이름': ['홍길동', '김철수', '홍길동', '이영희', '김철수'],
    '나이': [25, 30, 25, 35, 30],
    '도시': ['서울', '부산', '서울', '대구', '부산']
}
df = pd.DataFrame(data)

print(df.duplicated())        # 중복 행 True/False
print(df.duplicated().sum())  # 중복 행 개수: 2
```

### 2. 중복 제거 (`drop_duplicates`)

```python
df.drop_duplicates()                          # 모든 열 기준, 첫 번째 유지
df.drop_duplicates(subset=['이름'])            # 특정 열 기준
df.drop_duplicates(keep='last')               # 마지막 행 유지
df.drop_duplicates(keep=False)                # 중복된 행 모두 제거
```

### `keep` 매개변수

| 값 | 설명 |
| --- | --- |
| `'first'` (기본) | 첫 번째 행 유지 |
| `'last'` | 마지막 행 유지 |
| `False` | 중복된 행 모두 제거 |
