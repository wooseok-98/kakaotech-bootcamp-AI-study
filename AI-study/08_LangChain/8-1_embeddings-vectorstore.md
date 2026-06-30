# 8-1. 임베딩과 벡터 저장소 (Embeddings · VectorStore · Chroma · FAISS)

텍스트를 **벡터로 변환(임베딩)** 하고, 이를 저장·검색하는 **벡터 저장소(VectorStore)**

> RAG의 검색 단계 핵심 — 의미가 비슷한 텍스트는 벡터 공간에서 가깝게 위치한다는 원리 이용

---

## Embeddings

텍스트를 의미를 담은 **고정 길이 실수 벡터**로 변환하는 모델

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vec = embeddings.embed_query("LangChain이란?")   # 단일 쿼리 → 벡터
print(len(vec))   # 1536 (모델별 차원 고정)

vecs = embeddings.embed_documents(["문서1", "문서2"])  # 여러 문서 → 벡터 리스트
```

| 메서드 | 용도 |
|---|---|
| `embed_query(text)` | 검색 쿼리 1개를 벡터로 |
| `embed_documents([...])` | 저장할 문서 여러 개를 벡터로 |

> 쿼리와 문서를 같은 임베딩 모델로 변환해야 비교가 의미 있음

---

## VectorStore Interface

임베딩 벡터를 저장하고 **유사도 기반으로 검색**하는 저장소의 공통 인터페이스

| 메서드 | 설명 |
|---|---|
| `from_documents(docs, embeddings)` | 문서를 임베딩해 저장소 생성 |
| `add_documents(docs)` | 문서 추가 |
| `similarity_search(query, k)` | 쿼리와 유사한 상위 k개 문서 검색 |
| `similarity_search_with_score()` | 유사도 점수와 함께 반환 |
| `as_retriever()` | Retriever로 변환 (체인 연결용) |

```python
results = vectorstore.similarity_search("질문", k=3)
for doc in results:
    print(doc.page_content)
```

---

## Chroma

오픈소스 **로컬 벡터 DB** — 디스크에 영구 저장(persist) 가능, 간편한 설정

```python
from langchain_chroma import Chroma

# 문서로부터 생성 + 디스크 저장
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

# 저장된 DB 다시 불러오기
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
```

> `pip install langchain-chroma` 필요

---

## FAISS

Facebook이 개발한 **고속 유사도 검색 라이브러리** — 대용량·메모리 기반 검색에 강점

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)

# 로컬 파일로 저장 / 로드
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
```

> `pip install faiss-cpu` 필요

### Chroma vs FAISS

| 비교 | Chroma | FAISS |
|---|---|---|
| 성격 | 벡터 DB (메타데이터·필터링 내장) | 검색 라이브러리 (속도 특화) |
| 영구 저장 | `persist_directory` 자동 | `save_local` 수동 |
| 대규모 성능 | 보통 | 매우 빠름 |
| 사용 편의 | 높음 | 인덱스 직접 관리 |

> 학습·프로토타입은 Chroma, 대용량 고속 검색은 FAISS가 무난
