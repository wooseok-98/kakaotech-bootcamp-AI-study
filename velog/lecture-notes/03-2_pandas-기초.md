<!-- 📝 글 소개 (출간 시 입력): Pandas의 두 핵심 구조 Series와 DataFrame의 생성·접근·연산 정리 -->

# 🎈 Pandas 기초 (Series · DataFrame)

**Pandas**는 행과 열로 이루어진 표 형태 데이터를 다루는 파이썬 라이브러리다. **NumPy를 기반**으로 하며, 그 위에 표 데이터 관리·분석 기능을 얹었다.

핵심 구조는 두 가지다. **Series**는 1차원(열 하나), **DataFrame**은 2차원(표 전체)이며, <span style="color:red">DataFrame의 각 열이 곧 Series</span>다.

```python
import pandas as pd
df = pd.DataFrame({'이름': ['홍길동', '김철수'], '나이': [25, 30]})
```

___

## 📌 Series — 1차원

**인덱스를 가지는 1차원 배열**이다. 리스트와 달리 각 요소에 **레이블 인덱스**로 접근할 수 있다.

```python
s = pd.Series([100, 200, 300], index=['x', 'y', 'z'])
s['y']              # 200 (레이블로 접근)
s.values, s.index, s.dtype, s.size   # 기본 속성

s + 10              # 벡터 연산 (전체에 적용)
s[s > 150]          # 조건 필터링
```

결측치와 슬라이싱에서 주의할 점이 있다.

```python
s = pd.Series([1, 2, None, 4])
s.isnull()          # NaN 여부
s.fillna(0)         # NaN 대체

s2 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
s2['b':'d']         # ['b','c','d'] 모두 포함
```

> **레이블 슬라이싱은 끝 값을 포함**한다. NumPy의 정수 슬라이싱(`end` 미포함)과 반대라 헷갈리기 쉽다. 또한 문자열은 `object` 타입으로 저장된다.

___

## 📌 DataFrame — 2차원

**행과 열로 구성된 2차원 테이블**이다. 엑셀 시트나 DB 테이블과 유사하다.

```python
df = pd.DataFrame({'이름': ['홍길동', '김철수', '박영희'],
                   '나이': [25, 30, 28],
                   '성별': ['남', '남', '여']})
```

데이터를 받으면 **전체 구조부터 파악**하는 함수들을 먼저 쓴다.

```python
df.head()        # 처음 5행
df.shape         # (행, 열)
df.info()        # 타입·결측치 요약
df.describe()    # 수치형 통계 요약 (평균·표준편차·최대 등)
```

___

## 📌 데이터 접근 — loc vs iloc

DataFrame 접근에서 가장 중요한 구분이다.

| 방식 | 기준 | 예시 |
| --- | --- | --- |
| **`loc`** | 레이블(이름) | `df.loc['홍길동', '나이']` |
| **`iloc`** | 정수 위치 | `df.iloc[0, 1]` |

```python
df['이름']              # 열 선택 (Series)
df[['이름', '나이']]     # 여러 열 (DataFrame)

df.iloc[0]              # 첫 번째 행 (정수 위치)
df = df.set_index('이름')
df.loc['홍길동']         # '홍길동' 행 (레이블)
```

> 인덱스를 문자열로 지정하면 `df[0]` 같은 정수 접근은 에러가 난다. **"이름으로 찾으면 `loc`, 순서로 찾으면 `iloc`"**로 구분하면 헷갈리지 않는다.

___

## 📌 수정·연산·정렬

```python
df.loc['홍길동', '나이'] = 26          # 값 수정
df['총점'] = df['국어'] + df['영어']   # 기존 열 연산으로 새 열
df.drop('총점', axis=1, inplace=True) # 열 삭제

df[df['국어'] >= 90]                  # 조건으로 행 추출
df.sort_values(by='나이', ascending=False)  # 정렬
df.groupby('성별')['국어'].mean()     # 그룹별 집계
```

> `inplace=True`는 새 객체를 반환하지 않고 **원본을 직접 수정**한다. 생략하면 결과를 반환만 하므로 `df = df.drop(...)`처럼 다시 할당해야 한다.

___

## 📌 결측치와 파일 입출력

```python
df.isnull().sum()             # 열별 결측치 개수
df.dropna(inplace=True)       # 결측치 있는 행 제거
df.fillna(0, inplace=True)    # 결측치를 0으로 채움

df.to_csv('data.csv', index=False)   # 저장 (index=False로 인덱스 제외)
df = pd.read_csv('data.csv')          # 로딩
```
