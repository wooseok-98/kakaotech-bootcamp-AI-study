# 7-1. 텍스트 분할 (Text Splitter)

긴 문서를 검색·임베딩에 적합한 **작은 청크(chunk)** 로 나누는 도구

> 임베딩 모델·LLM의 입력 토큰 한도와, 검색 정확도(작은 단위일수록 관련성 높음) 때문에 분할 필요

---

## Text Splitter

문서를 일정 크기로 자르되, **청크 간 일부를 겹치게(overlap)** 해 문맥 손실을 줄이는 도구

| 매개변수 | 설명 |
|---|---|
| `chunk_size` | 청크 하나의 최대 크기 |
| `chunk_overlap` | 인접 청크 간 겹치는 크기 (문맥 연속성 유지) |
| `length_function` | 크기 측정 함수 (기본: `len`, 글자 수) |

> overlap이 너무 작으면 경계 정보 손실, 너무 크면 중복·비용 증가 — 보통 chunk_size의 10~20%

---

## RecursiveCharacterTextSplitter

여러 구분자를 **우선순위대로 시도**하며 의미 단위를 최대한 보존하는 분할기 — **가장 권장**

> `\n\n`(문단) → `\n`(줄) → ` `(단어) → `""`(글자) 순으로 시도, 청크가 chunk_size 이하가 될 때까지 재귀 분할

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_text(long_text)        # 문자열 분할
docs = splitter.split_documents(documents)     # Document 분할 (metadata 유지)
```

> 문단·문장 경계를 최대한 지키므로 일반 텍스트에 가장 적합

---

## CharacterTextSplitter

**단일 구분자** 하나만 기준으로 분할하는 가장 단순한 분할기

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",      # 이 구분자로만 분할
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_text(text)
```

> 구분자 기준으로 자른 뒤 합치므로, chunk_size를 넘는 덩어리가 생길 수 있음

---

## TokenTextSplitter

**토큰 수** 기준으로 분할 — LLM의 실제 토큰 한도에 정확히 맞춤

```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=256,        # 토큰 단위
    chunk_overlap=20,
)
chunks = splitter.split_text(text)
```

> 글자 수가 아닌 토큰 수로 자르므로 모델 컨텍스트 한도 관리에 유리 (tiktoken 기반)

### 분할기 비교

| 분할기 | 기준 | 특징 |
|---|---|---|
| `RecursiveCharacterTextSplitter` | 다중 구분자(재귀) | 의미 보존 최고, 기본 선택 |
| `CharacterTextSplitter` | 단일 구분자 | 단순, 크기 초과 가능 |
| `TokenTextSplitter` | 토큰 수 | 모델 토큰 한도에 정확 |
