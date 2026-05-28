# 3-3. 병합 (Merging)

**여러 데이터프레임을 공통 열 또는 인덱스를 기준으로 결합하는 과정**

SQL의 JOIN 연산과 유사

---

## 병합 종류

| 유형 | 설명 |
| --- | --- |
| **Inner Join** | 공통 키가 있는 행만 포함 |
| **Outer Join** | 모든 행 포함, 불일치 값은 NaN |
| **Left Join** | 왼쪽 DataFrame 전체 유지 |
| **Right Join** | 오른쪽 DataFrame 전체 유지 |

---

## 기본 문법

```python
pd.merge(left, right, on='기준 열', how='병합 방식')
```

---

## 예제 데이터

```python
import pandas as pd

customers = pd.DataFrame({
    '고객ID': [1, 2, 3],
    '이름': ['홍길동', '김철수', '이영희']
})

purchases = pd.DataFrame({
    '고객ID': [2, 3, 4],
    '구매액': [10000, 20000, 30000]
})
```

---

## 1. Inner Join

```python
pd.merge(customers, purchases, on='고객ID', how='inner')
#    고객ID   이름    구매액
# 0     2  김철수  10000
# 1     3  이영희  20000
```

---

## 2. Outer Join

```python
pd.merge(customers, purchases, on='고객ID', how='outer')
#    고객ID   이름      구매액
# 0     1  홍길동      NaN
# 1     2  김철수  10000.0
# 2     3  이영희  20000.0
# 3     4   NaN  30000.0
```

---

## 3. Left / Right Join

```python
pd.merge(customers, purchases, on='고객ID', how='left')   # 왼쪽 전체 유지
pd.merge(customers, purchases, on='고객ID', how='right')  # 오른쪽 전체 유지
```

---

## 4. 여러 열 기준 병합

```python
pd.merge(df1, df2, on=['고객ID', '도시'], how='inner')
```

---

## 5. 인덱스 기준 병합

```python
pd.merge(df1, df2, left_index=True, right_index=True, how='outer')
```

---

## 6. 열 이름 충돌 처리 (`suffixes`)

두 DataFrame에 같은 열 이름이 있을 때 접미사로 구분

```python
pd.merge(df1, df2, on='고객ID', suffixes=('_기존', '_신규'))
# 이름_기존, 이름_신규 처럼 생성됨
# on으로 지정한 열(조인 키)은 하나로 합쳐져 suffix 미적용
```

---

## 7. `concat()` vs `merge()`

| 구분 | 설명 |
| --- | --- |
| `merge()` | 공통 키 기준으로 병합 — 일치 여부 판단 |
| `concat()` | 단순 연결 (위/아래, 좌/우) — 키 없이 이어 붙임 |

```python
# 행 방향 연결
pd.concat([df1, df2], ignore_index=True)
```
