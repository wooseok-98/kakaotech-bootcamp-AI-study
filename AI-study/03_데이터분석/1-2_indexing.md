# 1-2. 인덱싱 (Indexing)

배열의 **특정 요소에 접근하거나 수정하는 방법**

---

## 1. 정수 인덱싱

개별 요소에 직접 접근. 0부터 시작, 음수는 끝에서부터

```python
import numpy as np

arr = np.array([10, 20, 30, 40])
print(arr[0])   # 10 (첫 번째)
print(arr[-1])  # 40 (마지막)

# 2차원 — [행, 열]
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(matrix[1, 2])  # 6 (2행 3열)
```

---

## 2. 슬라이싱

`start:end:step` 형식으로 범위 선택 — end는 포함되지 않음

```python
arr = np.array([10, 20, 30, 40, 50])

print(arr[1:4])  # [20 30 40]  — 1번~3번 인덱스
print(arr[:3])   # [10 20 30]  — 처음~2번 인덱스
print(arr[::2])  # [10 30 50]  — 0, 2, 4번 (2칸씩)

# 2차원에서 행/열 선택
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix[0])       # [1 2 3]   — 1행 전체
print(matrix[:, 1])    # [2 5 8]   — 모든 행의 2열
print(matrix[0:2, 1:]) # [[2 3] [5 6]]  — 1~2행의 2열 이후
```

---

## 3. 불리언 인덱싱

조건식이 True인 요소만 선택 — 데이터 필터링에 주로 사용

```python
data = np.array([5, 10, 15, 20, 25])

print(data[data > 10])      # [15 20 25]
print(data[data % 2 == 0])  # [10 20]

# 여러 조건 조합 (&: AND, |: OR)
print(data[(data > 10) & (data < 25)])  # [15 20]
```

---

## 4. 팬시 인덱싱

리스트나 배열로 여러 인덱스를 한 번에 선택

```python
arr = np.array([10, 20, 30, 40, 50])

print(arr[[0, 2, 4]])  # [10 30 50]

# 2차원에서 특정 행 선택
matrix = np.array([[1, 2], [3, 4], [5, 6]])
print(matrix[[0, 2]])
# [[1 2]
#  [5 6]]
```

---

## 5. 값 수정

```python
arr = np.array([10, 20, 30, 40])

arr[2] = 99          # 단일 요소 수정
arr[1:3] = [55, 77]  # 범위 수정
arr[arr < 50] = 0    # 조건으로 수정
```

---

## 인덱스 위치 찾기

```python
arr = np.array([5, 15, 8, 20, 3, 12])

# 조건을 만족하는 요소의 인덱스 반환
indices = np.where(arr > 10)
print(indices)       # (array([1, 3, 5]),)
print(arr[indices])  # [15 20 12]
```
