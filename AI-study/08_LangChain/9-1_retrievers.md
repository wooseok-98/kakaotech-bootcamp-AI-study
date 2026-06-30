# 9-1. 검색 추상화와 고급 Retrieval (Retriever · Ensemble · Multi-Query · Compression · Parent Document)

쿼리에 관련된 문서를 가져오는 **Retriever** 와, 검색 품질을 높이는 고급 검색 기법들

> 단순 벡터 검색의 한계(질문 표현 차이, 청크 크기 문제 등)를 보완하는 전략들

---

## Retriever

쿼리를 받아 **관련 Document 리스트를 반환**하는 검색 추상화 인터페이스

> VectorStore보다 상위 개념 — 벡터 검색뿐 아니라 키워드 검색, DB 검색 등도 동일 인터페이스로 추상화

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",       # 또는 "mmr"
    search_kwargs={"k": 4},
)
docs = retriever.invoke("LangChain이 뭐야?")
```

| search_type | 설명 |
|---|---|
| `similarity` | 유사도 상위 k개 (기본) |
| `mmr` | 관련성 + 다양성 균형 (중복 결과 감소) |

> Retriever는 Runnable이므로 LCEL 체인에 바로 연결 가능

---

## Ensemble Retriever

**여러 Retriever의 결과를 결합**해 가중치로 재순위화 (하이브리드 검색)

> 대표적으로 키워드 기반(BM25) + 의미 기반(벡터)을 합쳐 서로의 약점을 보완

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(docs)       # 키워드 검색
vector_ret = vectorstore.as_retriever()          # 의미 검색

ensemble = EnsembleRetriever(
    retrievers=[bm25, vector_ret],
    weights=[0.4, 0.6],                          # 결과 결합 가중치
)
```

| 검색 방식 | 강점 | 약점 |
|---|---|---|
| BM25 (키워드) | 정확한 단어 일치 | 동의어·문맥 약함 |
| Vector (의미) | 의미 유사성 | 고유명사·정확 매칭 약함 |

---

## Multi-Query Retriever

LLM으로 **원본 질문을 여러 변형 질문으로 확장**해 검색 — 표현 차이로 인한 누락 방지

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm,
)
# "LangChain 장점?" → 여러 표현으로 자동 생성 후 각각 검색, 결과 통합
```

> 질문 한 개의 표현에만 의존하지 않아 재현율(recall) 향상

---

## Contextual Compression

검색된 문서에서 **질문과 무관한 부분을 압축·필터링**해 핵심만 전달

> 청크에 노이즈가 섞여 있을 때, LLM에 넘기기 전 관련 내용만 추려 토큰 절약·정확도 향상

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)    # 관련 문장만 추출

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(),
)
```

| 압축기 | 동작 |
|---|---|
| `LLMChainExtractor` | LLM이 관련 문장만 추출 |
| `LLMChainFilter` | 관련 없는 문서 전체를 제거 |
| `EmbeddingsFilter` | 임베딩 유사도 기준으로 필터링 (LLM 호출 없음, 저렴) |

---

## Parent Document Retriever

**작은 청크로 검색하되, 응답에는 그 청크가 속한 큰 부모 문서를 반환**

> 검색 정확도(작은 청크)와 문맥 충분성(큰 문서)을 동시에 확보하는 절충안

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,                 # 작은 청크 임베딩 저장
    docstore=InMemoryStore(),                # 부모 문서 저장
    child_splitter=child_splitter,           # 검색용 작은 청크
    parent_splitter=parent_splitter,         # 반환용 큰 청크
)
retriever.add_documents(docs)
```

| 단위 | 역할 |
|---|---|
| Child (작은 청크) | 정확한 검색 매칭 |
| Parent (큰 문서) | 충분한 문맥을 LLM에 제공 |

### 고급 Retrieval 요약

| 기법 | 해결하는 문제 |
|---|---|
| Ensemble | 단일 검색 방식의 약점 보완 |
| Multi-Query | 질문 표현 차이로 인한 누락 |
| Contextual Compression | 검색 결과의 노이즈 |
| Parent Document | 청크 크기의 정확도-문맥 트레이드오프 |
