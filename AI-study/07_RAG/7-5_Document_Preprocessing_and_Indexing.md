# 문서 정제와 인덱싱 (Document Preprocessing & RAG Indexing)

## 개요

문서를 로딩한 이후 바로 임베딩하는 것이 아니라, 노이즈를 제거하고 검색 가능한 형태로 가공하는 과정이 필요합니다.

전체 흐름은 다음과 같습니다.

```text
Document Loading
    ↓
Text Preprocessing
    ↓
Chunking
    ↓
Embedding
    ↓
Vector Store
```

---

# 1. Text Preprocessing

## 정의

Text Preprocessing은 문서에서 추출한 텍스트를 청킹 및 임베딩 전에 정리하는 과정입니다.

주요 목적은 검색 품질을 떨어뜨리는 노이즈를 제거하는 것입니다.

---

## 왜 필요한가?

문서에서 추출한 텍스트에는 다음과 같은 불필요한 정보가 포함될 수 있습니다.

- 페이지 번호
- 반복 헤더
- 반복 푸터
- 제어 문자
- 과도한 공백
- 불필요한 빈 줄

예시

```text
스타트업코드 내부 문서

- 1 -

어댑터즈는 스타트업코드에서 제공하는 서비스입니다.
```

전처리 후

```text
어댑터즈는 스타트업코드에서 제공하는 서비스입니다.
```

---

## 주요 전처리 작업

| 작업 | 설명 |
|--------|--------|
| 유니코드 정규화 | 문자 표현 통일 |
| 제어 문자 제거 | 특수 제어 문자 제거 |
| 공백 정규화 | 연속 공백 제거 |
| 빈 줄 정리 | 과도한 줄바꿈 제거 |
| 헤더 제거 | 반복 헤더 제거 |
| 푸터 제거 | 페이지 번호 제거 |

---

## 대표 전처리 함수

```python
import re
import unicodedata

def preprocess_text(text):

    text = unicodedata.normalize("NFKC", text)

    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\xad]",
        "",
        text
    )

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
```

---

## 실무 팁

과도한 전처리는 오히려 정보 손실을 유발합니다.

예를 들어:

- 숫자 제거 ❌
- 코드 기호 제거 ❌
- 수식 제거 ❌

RAG에서는 의미 보존이 가장 중요합니다.

---

# 2. RAG Indexing

## 정의

RAG Indexing은 문서를 검색 가능한 상태로 만드는 전체 과정입니다.

```text
문서 로딩
→ 전처리
→ 청킹
→ 임베딩
→ 벡터 DB 저장
```

---

## 왜 필요한가?

사용자 질문이 들어왔을 때

```text
질문
 ↓
검색
 ↓
관련 문서 검색
 ↓
LLM 답변 생성
```

이 과정이 즉시 수행되기 위해서는 문서가 미리 벡터 DB에 저장되어 있어야 합니다.

---

## Indexing 파이프라인

| 순서 | 단계 |
|--------|--------|
| 1 | Document Loading |
| 2 | Text Preprocessing |
| 3 | Chunking |
| 4 | Embedding |
| 5 | Vector Store |

---

## Indexing vs Retrieval

| 항목 | Indexing | Retrieval |
|--------|--------|--------|
| 시점 | 사전 처리 | 질문 시 |
| 입력 | 문서 | 사용자 질문 |
| 출력 | 벡터 DB | 관련 문서 |
| 빈도 | 비정기 | 실시간 |

Indexing이 완료되어야 Retrieval이 가능합니다.

---

## 기본 청킹 함수

```python
def chunk_text(
    text,
    chunk_size=500,
    chunk_overlap=50
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += (
            chunk_size - chunk_overlap
        )

    return chunks
```

---

## 통합 인덱싱 과정

### 1단계 문서 로딩

```text
PDF
HTML
JSON
Markdown
```

### 2단계 전처리

```text
공백 정리
제어 문자 제거
유니코드 정규화
```

### 3단계 청킹

```text
500자 단위 분할
50자 중첩
```

### 4단계 임베딩

```python
SentenceTransformer
```

### 5단계 저장

```python
ChromaDB
```

---

## 초기 인덱싱

최초 구축 시

```text
전체 문서
 ↓
전체 인덱싱
 ↓
벡터 DB 저장
```

한 번만 수행합니다.

---

## 재인덱싱 (Incremental Indexing)

문서가 변경된 경우

```text
변경 문서 감지
 ↓
기존 벡터 삭제
 ↓
재임베딩
 ↓
재저장
```

전체 문서를 다시 처리하지 않습니다.

---

## 실무 팁

- 문서 로딩과 전처리는 분리
- 청킹 전략은 문서 특성에 따라 변경
- 메타데이터 반드시 저장
- 증분 인덱싱 사용
- ChromaDB 같은 벡터 DB 활용

---

# 정리

| 기술 | 역할 |
|--------|--------|
| Text Preprocessing | 노이즈 제거 |
| Chunking | 문서 분할 |
| Embedding | 벡터 변환 |
| Vector Store | 벡터 저장 |
| RAG Indexing | 전체 검색 준비 과정 |

다음 단계는 Retrieval이며, 저장된 벡터에서 질문과 가장 유사한 문서를 검색하는 과정입니다.
