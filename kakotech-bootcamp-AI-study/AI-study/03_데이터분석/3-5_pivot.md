# 3-5. 피벗 (Pivot)

**데이터를 특정 기준에 따라 재구성하여 행과 열을 재배치하는 과정**

| 방식 | 설명 |
| --- | --- |
| `pivot()` | 중복 없는 경우에만 사용. 단순 재구성 |
| `pivot_table()` | 중복 있어도 사용 가능. 집계 함수 적용 가능 |

---

## 구성 요소

| 매개변수 | 설명 |
| --- | --- |
| `index` | 새로운 행 인덱스로 설정할 열 |
| `columns` | 새로운 열로 설정할 열 |
| `values` | 값으로 채울 열 |

---

## 기본 문법

```python
df.pivot(index='날짜', columns='제품', values='판매량')
df.pivot_table(index='날짜', columns='제품', values='판매량', aggfunc='sum', fill_value=0)
```

---

## 1. 단순 피벗

```python
import pandas as pd

data = {
    '날짜': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
    '제품': ['A', 'B', 'A', 'B'],
    '판매량': [100, 200, 150, 250]
}
df = pd.DataFrame(data)

df_pivot = df.pivot(index='날짜', columns='제품', values='판매량')
# 제품            A    B
# 날짜
# 2024-01-01  100  200
# 2024-01-02  150  250
```

---

## 2. 다중 인덱스 피벗

```python
df.pivot(index=['날짜', '카테고리'], columns='제품', values='판매량')
```

---

## 3. `pivot_table()` — 중복 데이터 처리

```python
df.pivot_table(
    index='날짜',
    columns='제품',
    values='판매량',
    aggfunc='sum',   # sum, mean, count 등
    fill_value=0     # NaN을 0으로 채움
)
```

---

## 4. 여러 집계 함수 동시 적용

```python
df.pivot_table(
    index='날짜',
    columns='제품',
    values=['판매량', '이익'],
    aggfunc=['sum', 'mean']
)
```

---

## `pivot()` vs `pivot_table()` 차이

| | `pivot()` | `pivot_table()` |
| --- | --- | --- |
| 중복 처리 | 불가 (오류 발생) | 가능 (집계 함수 적용) |
| 결측치 처리 | 없음 | `fill_value`로 처리 |
| 집계 함수 | 없음 | `aggfunc`으로 지정 |
| 실무 활용 | 단순 재구성 | 더 많이 사용 |
