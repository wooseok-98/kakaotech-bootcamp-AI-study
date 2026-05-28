# 1-1. NumPy 기본

대규모 다차원 배열 및 행렬 연산을 위한 고성능 수학 함수와 도구를 제공하는 파이썬 라이브러리

파이썬 기본 리스트보다 빠르고 메모리 효율적으로 수치 데이터를 처리할 수 있음 (C 기반으로 구현)

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a)  # [1 2 3 4 5]
```

---

## 차원 (Dimension)

배열을 구성하는 **축(axis)의 개수** — `.ndim`으로 확인

| 차원 | 명칭 | 예시 |
| --- | --- | --- |
| 0차원 | 스칼라 | `np.array(42)` |
| 1차원 | 벡터 | `np.array([1, 2, 3])` |
| 2차원 | 행렬 | `np.array([[1, 2], [3, 4]])` |
| 3차원 | 텐서 | `np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])` |

차원 수는 **대괄호의 중첩 수준**으로 결정됨

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.ndim)  # 2
```

### 차원 변경

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# reshape - 원소 개수 유지하면서 형태 변경
reshaped = arr.reshape(2, 3)
print(reshaped.ndim)  # 2

# newaxis - 차원 추가
v = np.array([1, 2, 3])
col = v[:, np.newaxis]  # (3,) → (3, 1) 열 벡터
row = v[np.newaxis, :]  # (3,) → (1, 3) 행 벡터
```

---

## Shape

배열의 **각 차원별 요소 개수를 나타내는 튜플** — `.shape`으로 확인

```python
array = np.array([[1, 2, 3], [4, 5, 6]])
print(array.shape)  # (2, 3) → 2행 3열
print(array.ndim)   # 2 → 차원 수
```

> ndim은 "몇 차원인지", shape는 "각 차원에 요소가 몇 개인지"

### 형태 변경 방법

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# reshape - 새 배열 반환, 원소 개수 동일해야 함
print(arr.reshape(2, 3).shape)    # (2, 3)
print(arr.reshape(3, 2, 1).shape) # (3, 2, 1)

# flatten - 다차원 → 1차원, 새 배열 반환 (원본 영향 없음)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.flatten())  # [1 2 3 4 5 6]

# ravel - 다차원 → 1차원, 원본 참조 반환 (수정 시 원본도 변경됨)
raveled = matrix.ravel()
raveled[0] = 99
print(matrix)  # [[99 2 3] [4 5 6]] ← 원본 변경됨

# transpose - 행/열 교환
print(matrix.transpose())

# resize - 원본 자체를 변경, 부족한 값은 0으로 채움
arr2 = np.array([1, 2, 3, 4])
arr2.resize(2, 3)
print(arr2)
# [[1 2 3]
#  [4 0 0]]
```

---

## 데이터 타입 (Data Type)

배열의 각 요소가 가질 수 있는 **값의 유형을 지정하는 속성**

NumPy 배열은 모든 요소가 동일한 데이터 타입을 가져야 함 (Python 리스트와의 차이점)

| 타입 | 설명 | 예시 |
| --- | --- | --- |
| `int8 / int16 / int32 / int64` | 정수 (뒤 숫자 = 비트 수) | `-128 ~ 127` (int8 기준) |
| `uint8 / uint16 ...` | 부호 없는 정수 (0 이상만) | `0 ~ 255` (uint8 기준) |
| `float16 / float32 / float64` | 부동소수점 | `1.5, 3.14` |
| `bool_` | True / False | `True, False` |
| `str_` | 고정 길이 유니코드 문자열 | `<U6` (최대 6자) |
| `complex64 / complex128` | 복소수 (실수 + 허수) | `2 + 3j` |

```python
# 타입 확인
arr = np.array([10, 20, 30])
print(arr.dtype)  # int64

# 타입 지정해서 생성
int_arr   = np.array([1, 2, 3], dtype=np.int32)
float_arr = np.array([1.5, 2.3], dtype=np.float64)
str_arr   = np.array(["apple", "banana"], dtype=np.str_)

# 메모리 관련 속성
arr16 = np.array([10, 20, 30], dtype=np.int16)
print(arr16.itemsize)  # 2 → 각 요소당 바이트 수
print(arr16.nbytes)    # 6 → 배열 전체 바이트 수 (3개 × 2바이트)

# 타입 변환 — astype()
float_arr2 = int_arr.astype(np.float64)
print(float_arr2.dtype)  # float64
# 범위가 작은 타입으로 변환 시 오버플로우 주의
```
