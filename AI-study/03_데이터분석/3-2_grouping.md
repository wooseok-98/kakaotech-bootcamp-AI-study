# 3-2. 그룹화 (Grouping)

**데이터를 특정 기준에 따라 그룹화하여 집계, 변환, 필터링 등의 연산을 수행하는 기능**

"분할(Split) → 적용(Apply) → 결합(Combine)" 패턴으로 동작

---

## 기본 문법

```python
import pandas as pd

data = {
    '이름': ['홍길동', '김철수', '박영희', '이순신', '강감찬', '신사임당'],
    '부서': ['영업', '영업', '인사', '인사', 'IT', 'IT'],
    '급여': [5000, 5500, 4800, 5100, 6000, 6200]
}
df = pd.DataFrame(data)

# 부서별 급여 평균
df.groupby('부서')['급여'].mean()
# IT    6100.0
# 영업    5250.0
# 인사    4950.0
```

> DataFrame 형태로 받으려면 `reset_index()` 또는 `as_index=False` 사용

---

## 1. 단일 열 기준 그룹화

```python
df.groupby('부서')['급여'].sum()   # 부서별 합계
df.groupby('부서')['급여'].mean()  # 부서별 평균
```

---

## 2. 여러 열 기준 그룹화

```python
df.groupby(['부서', '이름'])['급여'].sum()
```

---

## 3. `agg()` — 여러 집계 함수 동시 적용

```python
df.groupby('부서')['급여'].agg(['sum', 'mean', 'max', 'min'])
#       sum    mean   max   min
# IT  12200  6100.0  6200  6000
# 영업  10500  5250.0  5500  5000
# 인사   9900  4950.0  5100  4800
```

---

## 4. `filter()` — 그룹 단위 필터링

조건을 만족하는 그룹에 속한 행 전체를 반환

```python
df.groupby('부서').filter(lambda x: x['급여'].sum() > 10000)
```

---

## 5. `transform()` — 그룹별 값을 원래 행에 매핑

원본 크기를 유지하면서 그룹별 계산 결과를 각 행에 적용

```python
df['급여_평균'] = df.groupby('부서')['급여'].transform('mean')
# 각 행에 해당 부서의 평균 급여가 채워짐
```

---

## 6. 사용자 정의 함수 적용

```python
def categorize(x):
    return '고임금' if x.mean() > 5000 else '저임금'

df['등급'] = df.groupby('부서')['급여'].transform(categorize)
```

---

## 7. `rank()` — 그룹별 순위

```python
df['순위'] = df.groupby('부서')['급여'].rank(ascending=False)
```
