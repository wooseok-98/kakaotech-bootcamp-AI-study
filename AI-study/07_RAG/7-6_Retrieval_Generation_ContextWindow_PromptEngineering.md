
# 검색과 생성 (Retrieval, Generation, Context Window, Prompt Engineering)

## RAG 전체 흐름

Indexing → Retrieval → Generation

---

# 1. Retrieval

## 정의

사용자 질문과 가장 유사한 문서 청크를 벡터 DB에서 검색하는 단계입니다.

### 동작 흐름

질문 → 질문 임베딩 → 유사도 검색 → Top-K 반환 → Generation 전달

## 왜 필요한가?

- 답변의 근거 확보
- 최신 정보 활용
- 환각 감소

## 핵심 요소

- 임베딩 모델
- 청킹 전략
- Top-K
- 메타데이터 필터

### 기본 코드

```python
query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)
```

---

# 2. Generation

## 정의

검색된 문서를 근거로 LLM이 최종 답변을 생성하는 단계입니다.

### 동작 흐름

검색 결과 → 프롬프트 구성 → LLM 호출 → 답변 생성

## 왜 필요한가?

검색 결과는 문서 조각일 뿐입니다.

LLM이 이를 자연어로 재구성하여 사용자에게 제공합니다.

### 기본 구조

```text
[문서]
검색 결과

[질문]
사용자 질문
```

---

# 3. Context Window

## 정의

LLM이 한 번에 처리할 수 있는 최대 토큰 수입니다.

### 포함 요소

- 시스템 프롬프트
- 검색 문서
- 사용자 질문
- 생성 답변

## 관리 방법

전체 토큰 수는 모델 한도를 넘으면 안 됩니다.

```text
시스템 프롬프트
+ 검색 문서
+ 질문
+ 답변 예산
≤ Context Window
```

### Top-K 영향

Top-K가 증가할수록 검색 문서 토큰 수가 증가합니다.

예시

- Chunk Size 500
- Top-K 3 → 약 1500 토큰
- Top-K 10 → 약 5000 토큰

---

# 4. Prompt Engineering for RAG

## 정의

검색된 문서를 근거로만 답변하도록 프롬프트를 설계하는 기법입니다.

## 목적

- 정확도 향상
- 환각 감소
- 답변 일관성 향상

## 권장 구조

1. 역할 지정
2. 근거 제한
3. 문서 삽입
4. 질문 삽입
5. 출력 형식 지정

### 예시

```text
당신은 사내 문서 QA 시스템입니다.

아래 문서만을 근거로 답하세요.

문서에 없는 내용은
'해당 정보를 찾을 수 없습니다'
라고 답하세요.

[문서 1]
...

[문서 2]
...

질문:
...
```

## 좋은 프롬프트 원칙

### 명확하게 지시

❌ 짧게 답해

✅ 3문장 이내로 답하세요

### 근거 제한

❌ 문서를 참고하여 답하세요

✅ 문서만을 근거로 답하세요

### 문서 번호 사용

[문서 1], [문서 2]

### 모를 때 행동 지정

문서에 없으면 찾을 수 없다고 답하도록 지정

### 출력 형식 지정

- 한국어
- 3문장 이내
- JSON 형식 등

---

# Retrieval vs Generation

| 항목 | Retrieval | Generation |
|------|------------|------------|
| 역할 | 문서 검색 | 답변 생성 |
| 입력 | 사용자 질문 | 질문 + 문서 |
| 출력 | 문서 청크 | 자연어 답변 |
| 모델 | 임베딩 모델 | LLM |

---

# 정리

| 기술 | 역할 |
|------|------|
| Retrieval | 관련 문서 검색 |
| Generation | 답변 생성 |
| Context Window | 토큰 한도 관리 |
| Prompt Engineering | 답변 품질 향상 |

좋은 RAG는

Indexing → Retrieval → Prompt Engineering → Generation

순서로 품질이 결정됩니다.
