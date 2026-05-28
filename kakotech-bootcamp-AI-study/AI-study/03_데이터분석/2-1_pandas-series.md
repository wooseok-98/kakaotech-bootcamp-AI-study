# 2-1. Pandas & 시리즈 (Series)

## Pandas 기본

구조화된 데이터의 조작과 분석을 위한 데이터프레임 및 시리즈 객체를 제공하는 파이썬 라이브러리

NumPy를 기반으로 하며, 행과 열로 이루어진 표 형태의 데이터를 다루는 데 최적화되어 있음

| 특징 | 설명 |
| --- | --- |
| **데이터 조작 및 변환** | 선택, 필터링, 정렬, 집계, 피벗, 결측치 처리, 데이터 형 변환 등 |
| **빠른 연산 속도** | NumPy 기반 벡터화 연산으로 반복문 없이 대량 데이터 처리 |
| **다양한 데이터 소스** | CSV, 엑셀, SQL, JSON 등 파일 형식 지원 |
| **시각화 연동** | Matplotlib, Seaborn 등과 원활하게 연동 |

> NumPy는 수치 계산·다차원 배열에 특화, Pandas는 그 위에 표 형태 데이터 관리와 분석 기능을 얹은 확장판

```python
import pandas as pd

print(pd.__version__)  # 2.2.2 (환경마다 다름)

# 간단한 예제
data = {
    '이름': ['홍길동', '김철수', '이영희'],
    '나이': [25, 30, 35],
    '도시': ['서울', '부산', '인천']
}
df = pd.DataFrame(data)
print(df)
```

---

## 시리즈 (Series)

**인덱스를 가지는 1차원 배열 형태의 데이터 구조**

리스트, 딕셔너리, NumPy 배열과 유사하지만, 고유한 인덱스를 통해 각 요소에 접근할 수 있음
DataFrame의 각 열(column)이 Series로 구성됨

### 주요 특징

| 특징 | 설명 |
| --- | --- |
| **인덱스 기반 1차원 구조** | 기본 정수 인덱스(0부터 시작) 또는 문자열, 날짜 등 레이블 인덱스 지정 가능 |
| **다양한 데이터 유형 지원** | 정수, 실수, 문자열, 불리언, 날짜 등 |
| **벡터 연산 지원** | NumPy와 유사한 요소별 연산 및 브로드캐스팅 |
| **결측치 처리** | `NaN` 자동 감지 및 처리 기능 내장 |

### 기본 속성

| 속성 | 설명 |
| --- | --- |
| `values` | 데이터 값을 ndarray 형식으로 반환 |
| `index` | 인덱스 반환 |
| `dtype` | 데이터 유형 반환 |
| `shape` | 크기(튜플) 반환 |
| `size` | 총 요소 수 반환 |
| `name` | Series 이름 설정 및 확인 |

### 데이터 타입

| 타입 | 예시 |
| --- | --- |
| `int64` | `pd.Series([1, 2, 3], dtype='int64')` |
| `float64` | `pd.Series([1.1, 2.2], dtype='float64')` |
| `object` | `pd.Series(['a', 'b'], dtype='object')` — 문자열은 object로 저장 |
| `bool` | `pd.Series([True, False], dtype='bool')` |
| `datetime64` | `pd.Series(pd.date_range('20240101', periods=3))` |

### 1. 생성

```python
import pandas as pd

# 리스트
s1 = pd.Series([1, 2, 3, 4])

# 딕셔너리 — 키가 인덱스로
s2 = pd.Series({'a': 10, 'b': 20, 'c': 30})

# 인덱스 직접 지정
s3 = pd.Series([100, 200, 300], index=['x', 'y', 'z'])
```

### 2. 기본 속성 확인

```python
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])

print(s.values)  # [10 20 30 40]
print(s.index)   # Index(['a', 'b', 'c', 'd'])
print(s.dtype)   # int64
print(s.shape)   # (4,)
print(s.size)    # 4

s.name = "점수"
```

### 3. 데이터 접근 및 수정

```python
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

print(s['a'])       # 10
s.index = ['x', 'y', 'z']  # 인덱스 변경
s['y'] = 50         # 값 수정
```

### 4. 연산

```python
s = pd.Series([1, 2, 3, 4])

print(s + 10)       # 모든 요소에 10 더하기
print(s * 2)        # 모든 요소를 2배
print(s[s > 2])     # 조건 필터링
```

### 5. 결측치 처리

```python
s = pd.Series([1, 2, None, 4])

print(s.isnull())   # NaN 여부 확인
print(s.fillna(0))  # NaN을 0으로 대체
```

### 6. 슬라이싱

```python
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])

print(s[s > 20])    # 20보다 큰 요소
print(s['b':'d'])   # 인덱스 범위 — end 포함
```
