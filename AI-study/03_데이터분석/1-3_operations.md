# 1-3. 연산 & 유니버설 함수 (Operation & Universal Function)

**연산**: 배열의 요소에 대해 수행되는 수학적 또는 논리적 계산

**유니버설 함수(UFuncs)**: 배열의 각 요소에 반복 수행되는 벡터화된 연산을 제공하는 함수 (`np.함수명()` 형태)

핵심 특징:
- **벡터화** — 반복문 없이 배열 전체에 동시 연산
- **요소별(Element-wise)** — 같은 위치의 요소끼리 계산
- **브로드캐스팅** — 크기가 다른 배열 간 연산을 자동으로 조정

---

## 1. 산술 연산

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)             # [5 7 9]
print(a - b)             # [-3 -3 -3]
print(a * b)             # [ 4 10 18]
print(a / b)             # [0.25 0.4 0.5]

# ufunc 함수 형태 (동일한 결과)
print(np.add(a, b))      # [5 7 9]
print(np.multiply(a, b)) # [ 4 10 18]
print(np.power(a, 2))    # [1 4 9]  — 거듭제곱
```

---

## 2. 비교 연산

```python
a = np.array([1, 2, 3, 4])
b = np.array([2, 2, 2, 2])

print(a > 2)             # [False False  True  True]
print(a == b)            # [False  True False False]
print(np.equal(a, b))    # [False  True False False]
print(np.greater(a, b))  # [False False  True  True]
```

---

## 3. 논리 연산

```python
data = np.array([10, 20, 30, 40])

print(np.logical_and(data > 15, data < 35))  # [False  True  True False]
print(np.logical_or(data < 15, data > 35))   # [ True False False  True]
print(np.logical_not(data > 20))             # [ True  True False False]
```

---

## 4. 통계 연산

```python
data = np.array([10, 20, 30, 40, 50])

print(np.mean(data))    # 30.0   — 평균
print(np.median(data))  # 30.0   — 중앙값
print(np.max(data))     # 50     — 최댓값
print(np.min(data))     # 10     — 최솟값
print(np.std(data))     # 14.14  — 표준편차
print(np.sum(data))     # 150    — 합계

# axis 기준 연산 (2차원 배열)
matrix = np.array([[3, 7, 2],
                   [8, 4, 6]])
print(np.max(matrix, axis=0))  # [8 7 6] — 열 기준 최댓값
print(np.max(matrix, axis=1))  # [7 8]   — 행 기준 최댓값
```

---

## 5. 브로드캐스팅

크기가 다른 배열 간 연산 시 작은 배열을 자동으로 확장

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])   # (2, 3)
vector = np.array([1, 2, 3])    # (3,)

print(matrix + vector)
# [[2 4 6]
#  [5 7 9]]   ← vector가 각 행에 더해짐

print(matrix + 10)
# [[11 12 13]
#  [14 15 16]]
```

> 브로드캐스팅 규칙: 오른쪽(마지막 축)부터 크기를 맞춤. 크기가 같거나 한쪽이 1이면 연산 가능

---

## 6. 선형대수 연산

```python
matrix = np.array([[1, 2], [3, 4]])
vector = np.array([2, 3])

print(np.transpose(matrix))    # 전치 — 행과 열 교환
print(np.dot(matrix, vector))  # 내적(행렬 곱) → [8 18]
print(np.linalg.inv(matrix))   # 역행렬
```

---

## 7. 삼각함수 & 지수/로그

```python
angles = np.array([0, np.pi/2, np.pi])
print(np.sin(angles))   # [0.  1.  0.]
print(np.cos(angles))   # [ 1.  0. -1.]
# e-16 같은 값은 부동소수점 오차 — 실제로는 0으로 해석

values = np.array([1, np.e, 10])
print(np.exp(values))    # e^x
print(np.log(values))    # 자연로그(밑 e)
print(np.log10(values))  # 상용로그(밑 10)
```

---

## 8. `out` 매개변수

새 배열을 만들지 않고 기존 배열에 결과 저장 → 메모리 절약

```python
a = np.array([1, 2, 3])
result = np.empty_like(a)

np.multiply(a, 10, out=result)
print(result)  # [10 20 30]
```
