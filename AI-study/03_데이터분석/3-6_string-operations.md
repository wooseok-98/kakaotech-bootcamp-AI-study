# 3-6. 문자열 처리 (String Operations)

**Series의 `.str` 접근자를 통해 문자열 데이터를 벡터화 방식으로 처리하는 기능**

`.str` 접근자를 사용하면 반복문 없이 배열 전체에 문자열 메서드를 적용할 수 있음

---

## 주요 메서드

| 메서드 | 설명 |
| --- | --- |
| `str.upper()` / `str.lower()` | 대소문자 변환 |
| `str.strip()` | 앞뒤 공백 제거 |
| `str.replace(a, b)` | 문자열 치환 |
| `str.contains(pat)` | 패턴 포함 여부 (True/False) |
| `str.startswith(s)` / `str.endswith(s)` | 시작/끝 문자 확인 |
| `str.split(sep)` | 구분자로 분리 |
| `str.len()` | 문자열 길이 |
| `str.extract(pat)` | 정규표현식으로 추출 |

---

## 1. 대소문자 변환

```python
import pandas as pd

s = pd.Series(['hello', 'WORLD', 'Pandas'])

print(s.str.upper())   # ['HELLO', 'WORLD', 'PANDAS']
print(s.str.lower())   # ['hello', 'world', 'pandas']
```

---

## 2. 공백 제거

```python
s = pd.Series(['  홍길동  ', ' 김철수', '이영희 '])

print(s.str.strip())    # 양쪽 공백 제거
print(s.str.lstrip())   # 왼쪽 공백 제거
print(s.str.rstrip())   # 오른쪽 공백 제거
```

---

## 3. 문자열 치환

```python
s = pd.Series(['서울시 강남구', '부산시 해운대구'])

print(s.str.replace('시', ''))
# ['서울 강남구', '부산 해운대구']
```

---

## 4. 패턴 포함 여부 — 필터링 활용

```python
df = pd.DataFrame({'이름': ['홍길동', '김철수', '김영희']})

print(df[df['이름'].str.contains('김')])        # '김' 포함
print(df[df['이름'].str.startswith('홍')])      # '홍'으로 시작
print(df[df['이름'].str.endswith('수')])        # '수'로 끝
```

---

## 5. 문자열 분리

```python
s = pd.Series(['홍길동-25-서울', '김철수-30-부산'])

# 구분자로 분리 → 리스트 반환
print(s.str.split('-'))

# 분리 후 DataFrame으로 확장
df_split = s.str.split('-', expand=True)
df_split.columns = ['이름', '나이', '도시']
print(df_split)
```

---

## 6. 문자열 길이

```python
s = pd.Series(['홍길동', '김철수', '이영희'])
print(s.str.len())   # [3, 3, 3]
```

---

## 7. 정규표현식으로 추출

```python
s = pd.Series(['홍길동(25세)', '김철수(30세)'])

# 괄호 안의 숫자 추출
print(s.str.extract(r'\((\d+)세\)'))
#     0
# 0  25
# 1  30
```
