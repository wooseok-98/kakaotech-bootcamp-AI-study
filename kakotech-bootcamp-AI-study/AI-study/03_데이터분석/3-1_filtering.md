# 3-1. 필터링 (Filtering)

**조건을 적용하여 데이터프레임이나 시리즈에서 특정 행이나 값을 선택하는 과정**

| 방법 | 설명 |
| --- | --- |
| `df[조건]` | 불리언 인덱싱 |
| `query()` | SQL 문자열 스타일 조건 |
| `isin()` | 특정 값 목록 포함 여부 |
| `str.contains()` | 문자열 패턴 포함 여부 |
| `isnull() / notnull()` | 결측치 여부 기반 필터링 |

---

## 1. 불리언 인덱싱

```python
import pandas as pd

data = {
    '이름': ['홍길동', '김철수', '박영희', '이순신', '강감찬'],
    '나이': [25, 30, 35, 40, 45],
    '도시': ['서울', '부산', '서울', '대구', '부산'],
    '점수': [85, 90, 75, 95, 80]
}
df = pd.DataFrame(data)

print(df[df['나이'] > 30])       # 단일 조건
print(df[df['점수'] >= 85])
```

---

## 2. 다중 조건 (`&`, `|`, `~`)

```python
# AND: 두 조건 모두 만족
print(df[(df['나이'] >= 30) & (df['점수'] > 80)])

# OR: 둘 중 하나 만족
print(df[(df['도시'] == '서울') | (df['점수'] >= 90)])

# NOT: 조건 반전
print(df[~(df['나이'] <= 40)])
```

---

## 3. `query()`

SQL 스타일로 조건을 문자열로 작성 — 조건이 복잡할 때 가독성 향상

```python
print(df.query("점수 > 85"))
print(df.query("나이 >= 30 and 도시 == '부산'"))
```

---

## 4. `isin()`

특정 값 목록에 해당하는 행 선택

```python
print(df[df['도시'].isin(['서울', '부산'])])
print(df[df['이름'].isin(['김철수', '이순신'])])
```

---

## 5. 문자열 필터링

```python
print(df[df['이름'].str.contains('김')])      # '김' 포함
print(df[df['도시'].str.startswith('부')])    # '부'로 시작
```

---

## 6. `apply()` + lambda

조건 결과를 새로운 열로 추가

```python
df['합격여부'] = df['점수'].apply(lambda x: '합격' if x >= 90 else '불합격')
```

---

## 7. 결측치 필터링

```python
print(df[df['점수'].isnull()])     # 결측치 있는 행
print(df[df['점수'].notnull()])    # 결측치 없는 행
```
