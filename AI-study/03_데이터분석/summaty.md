# Numpy / Pandas 핵심 한줄 정리

---

## Numpy 기본

| 개념 | 핵심 |
|------|------|
| Numpy | 파이썬 리스트보다 빠른 수치 연산을 위한 배열 라이브러리 (`import numpy as np`) |
| 차원 (Dimension) | 0차원=스칼라, 1차원=벡터, 2차원=행렬, 3차원 이상=텐서 — `ndim`으로 확인 |
| Shape | 각 차원의 크기를 나타내는 튜플 — `arr.shape` → `(행, 열)` |
| 데이터 타입 (dtype) | 배열 안 원소의 타입 — `arr.dtype`으로 확인, 생성 시 `dtype=np.float32` 지정 가능 |
| 인덱싱 (Indexing) | `arr[행, 열]` 또는 `arr[0:3, 1:4]` 슬라이싱 — 파이썬 리스트와 달리 뷰(view)를 반환하므로 수정 시 원본도 바뀜 |
| 연산 (Operation) | 같은 shape끼리 원소별 연산 자동 적용 — `+`, `-`, `*`, `/` 모두 브로드캐스팅 지원 |
| 유니버설 함수 (ufunc) | `np.sqrt()`, `np.exp()`, `np.sum()`, `np.mean()` 등 배열 전체에 한번에 적용되는 함수 |

---

## Pandas 기본

| 개념 | 핵심 |
|------|------|
| Pandas | 표 형태 데이터를 다루는 라이브러리 (`import pandas as pd`) |
| Series | 인덱스가 붙은 1차원 배열 — `pd.Series([1, 2, 3])`, 딕셔너리처럼 라벨로 접근 가능 |
| DataFrame | 인덱스와 컬럼명이 붙은 2차원 표 — `df['컬럼명']`으로 열 선택, `df.iloc[행번호]`로 행 선택 |

---

## 데이터 변형 및 처리

| 개념 | 설명 | 핵심 |
|------|------|------|
| 필터링 (Filtering) | 특정 조건을 만족하는 행만 골라내는 것 | 조건식으로 행을 선택 — `df[df['나이'] > 20]`, 여러 조건은 `&` `\|` 사용 |
| 그룹화 (Grouping) | 특정 컬럼 값이 같은 행끼리 묶어 집계하는 것 | `df.groupby('컬럼').집계함수()` — 카테고리별 합계/평균 등 집계 |
| 병합 (Merging) | 공통 키를 기준으로 두 DataFrame을 하나로 합치는 것 | `pd.merge(df1, df2, on='키컬럼')` — SQL의 JOIN과 동일한 개념 |
| 결측치 처리 (Missing Data) | 데이터에 빠진 값(NaN)을 찾아 제거하거나 채우는 것 | `df.isnull()`로 확인, `df.dropna()`로 제거, `df.fillna(값)`으로 대체 |
| 피벗 (Pivot) | 행과 열의 기준을 바꿔 데이터를 요약 표로 재구성하는 것 | `df.pivot_table(values, index, columns, aggfunc)` — 행/열 기준을 바꿔 요약 테이블 생성 |
| 중복 제거 (Duplicates) | 완전히 동일한 행이 여러 번 있을 때 하나만 남기는 것 | `df.duplicated()`로 중복 확인, `df.drop_duplicates()`로 제거 |
| 문자열 처리 (String) | 텍스트 컬럼에 포함/치환/분리 등 문자열 연산을 적용하는 것 | `df['컬럼'].str.메서드()` — `.str.contains()`, `.str.replace()`, `.str.split()` 등 벡터화 문자열 연산 |