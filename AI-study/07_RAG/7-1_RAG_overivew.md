# 8-1. RAG · Tokenization · Text Embedding · Dense Vector · Sparse Vector

---

## RAG (Retrieval-Augmented Generation)

**외부 데이터 저장소에서 관련 문서를 검색하고, 그 결과를 LLM 프롬프트에 삽입하여 근거 기반의 답변을 생성하는 아키텍처**

- RAG는 모델 이름이 아니라 **시스템 패턴(아키텍처)**
- 모델 자체를 재학습하는 방식이 아니라, 질문 시점에 필요한 정보를 찾아와 함께 사용

### 핵심 개념

| 개념 | 설명 |
| --- | --- |
| 파라메트릭 지식 (Parametric Knowledge) | LLM이 학습 과정에서 가중치 안에 압축된 지식 ? 추론 시점에 자동으로 갱신되지 않음 |
| 논-파라메트릭 지식 (Non-Parametric Knowledge) | 문서, 데이터베이스, API처럼 모델 밖에 따로 저장된 지식 ? RAG가 질문 시점에 검색해 사용 |
| Closed-Book QA | LLM이 학습된 지식만으로 답변하는 방식 |
| Open-Book QA | 외부 자료를 참고하여 답변하는 방식 ? RAG가 대표적인 구현 방식 |

### RAG 파이프라인 구성

RAG 파이프라인은 **인덱싱(Indexing)** 단계와 **질의(Query)** 단계로 나뉨

| 단계 | 구성 요소 | 역할 |
| --- | --- | --- |
| 인덱싱 | Document Loading | 원본 문서(PDF, HTML, TXT 등)를 시스템에 불러오는 단계 |
| 인덱싱 | Chunking | 긴 문서를 검색에 적합한 작은 단위(청크)로 분할 |
| 인덱싱 | Embedding | 텍스트 청크를 숫자 벡터로 변환 |
| 인덱싱 | Vector Store | 임베딩 벡터와 원본 텍스트를 저장하고 유사도 검색을 수행하는 저장소 |
| 질의 | Retrieval | 사용자 질문을 벡터화한 뒤 유사한 문서 청크를 검색 |
| 질의 | Augmentation | 검색된 청크를 프롬프트 템플릿에 삽입하여 LLM 입력 구성 |
| 질의 | Generation | LLM이 삽입된 문서를 근거로 답변 생성 |

> 인덱싱은 한 번만 하면 되는 것이 아니라, 문서가 변경될 때마다 다시 수행해야 함. 실무에서는 변경분만 처리하는 **증분 인덱싱(Incremental Indexing)** 사용

### 사용 이유

| RAG | 모델 재학습 없이 최신 정보와 내부 데이터를 반영한 답변을 생성하고, 환각을 줄이며, 답변 근거를 추적할 수 있는 신뢰성 높은 시스템을 구축하기 위해서 |
| --- | --- |

- LLM은 학습 시점 이후의 정보를 알 수 없음 → RAG로 최신 문서를 질문 시 참조
- 사내 문서, 고객 지원 이력 등 비공개 정보를 모델 가중치 변경 없이 유연하게 연결
- 검색된 문서를 출처로 제공할 수 있어 답변 근거 추적 가능

### 사용 방법

| RAG | 문서를 청크로 분할하고 임베딩하여 벡터 DB에 저장한 뒤, 사용자 질문과 유사한 문서를 검색하여 프롬프트에 삽입하고 LLM으로 답변을 생성 |
| --- | --- |

| 순서 | 단계 | 설명 |
| --- | --- | --- |
| 1 | 문서 로딩 | PDF, HTML, TXT 등 원본 문서를 시스템에 불러옴 |
| 2 | 텍스트 전처리 | 불필요한 태그, 특수문자, 빈 줄 등을 제거 |
| 3 | 청킹 | 문서를 일정 크기(예: 500~1000자)의 청크로 분할. 청크 간 겹침(overlap)을 두어 문맥 손실 최소화 |
| 4 | 임베딩 | 각 청크를 임베딩 모델로 벡터화 |
| 5 | 벡터 저장 | 생성된 벡터와 원본 텍스트를 벡터 DB(ChromaDB, FAISS 등)에 저장 |
| 6 | 질문 수신 | 사용자가 자연어로 질문 입력 |
| 7 | 질문 임베딩 | 인덱싱 때와 **동일한** 임베딩 모델로 질문 벡터화 |
| 8 | 유사 문서 검색 | 질문 벡터와 저장된 문서 벡터의 유사도를 계산하여 상위 k개 청크 검색 |
| 9 | 프롬프트 구성 | 검색된 청크를 프롬프트 템플릿에 삽입 |
| 10 | 답변 생성 | LLM이 프롬프트를 받아 문서 기반 답변 생성 |

### 코드 예시

```python
# 필요한 패키지 설치
!pip install -q sentence-transformers transformers torch numpy

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ===== 1. 문서 준비 =====
documents = [
    "Adapterz is a developer book serving service provided by Startupcode.",
    "Startupcode is a company specializing in developer education.",
    "RAG retrieves external data, inserts it into the LLM input, and then generates an answer.",
    "Fine-tuning directly modifies model weights to specialize in a domain.",
]

# ===== 2. 전처리 + 청킹 =====
def preprocess(text):
    return re.sub(r"\s+", " ", text).strip()

def split_into_chunks(text, chunk_size=120, overlap=30):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

all_chunks = []
for doc in documents:
    all_chunks.extend(split_into_chunks(preprocess(doc)))

# ===== 3. 임베딩 + 벡터 저장 =====
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_vectors = embedding_model.encode(all_chunks)

print(f"청크 수: {len(chunk_vectors)}, 벡터 차원: {chunk_vectors.shape[1]}")

# ===== 4. 유사 문서 검색 =====
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query, chunk_vectors, all_chunks, model, top_k=2):
    query_vec = model.encode(query)
    scores = [(i, cosine_similarity(query_vec, v), all_chunks[i]) for i, v in enumerate(chunk_vectors)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

# ===== 5. 프롬프트 구성 + 답변 생성 =====
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def rag_query(question):
    results = search(question, chunk_vectors, all_chunks, embedding_model)
    context = "\n".join([chunk for _, _, chunk in results])
    prompt = (
        f"Answer based only on the following documents.\n"
        f"If not found, say 'The information is not found in the documents.'\n\n"
        f"Documents:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    output = llm.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(output[0], skip_special_tokens=True), results

answer, results = rag_query("What is Adapterz?")
print(f"A: {answer}")
```

### 다른 기술과의 비교

| 기술 | 특징 |
| --- | --- |
| RAG | 외부 문서를 검색하여 프롬프트에 삽입. 모델 변경 없이 최신 정보 반영. 문서 교체만으로 업데이트 |
| Fine-tuning | 모델 가중치를 직접 수정. 도메인 특화 성능 향상. 데이터 변경 시 재학습 필요, GPU 비용 발생 |
| Long Context | 문서 전체를 컨텍스트 윈도우에 삽입. 구현이 단순하지만 문서가 많을수록 비용·지연 급증 |
| Prompt Engineering | 프롬프트 설계만으로 품질 개선. 대규모 외부 문서 자동 연동 불가 |

> **실무 패턴**: 기본 모델 + RAG 조합으로 먼저 시작 → 도메인 특화 추론이 필요할 때만 Fine-tuning 추가

### RAG에서 자주 발생하는 문제

| 문제 | 원인 | 해결 방법 |
| --- | --- | --- |
| 관련 없는 문서가 검색됨 | 청킹 방식 부적절, 임베딩 모델 성능 부족 | 청크 크기 조정, 더 나은 임베딩 모델, Hybrid Search, Reranking |
| 환각 (Hallucination) | 검색 문서가 틀리거나 부족 | "문서에 없으면 모른다고 답하라" 프롬프트 지시, 유사도 임계값 설정 |
| 중복 문서가 반복 검색됨 | 청크 간 overlap으로 유사 내용 중복 저장 | 중복 제거, 검색 결과 다양성 확보 |
| 청크 경계에서 문맥 끊김 | 긴 문서를 나누는 과정에서 중요 정보가 분리됨 | overlap 설정, Parent-Child Chunking |

---

## Tokenization (토큰화)

**텍스트를 모델이 처리할 수 있는 토큰(단어, 서브워드, 문자 등) 단위로 분리하는 과정**

- 컴퓨터는 문자열을 직접 처리할 수 없으므로, 숫자로 변환하기 위한 첫 단계
- 토큰은 토크나이저가 정한 최소 처리 단위 ? 단어보다 작을 수도, 같을 수도 있음

### Tokenization의 주요 방식

| 방식 | 분리 기준 | 예시 ("Startupcode is a company.") | 특징 |
| --- | --- | --- | --- |
| Word Tokenization | 공백·구두점 기준 단어 단위 분리 | ["Startupcode", "is", "a", "company."] | 구현 단순. OOV(미등록 단어) 문제 발생 가능 |
| Subword Tokenization | 빈도 기반 부분 문자열 단위 분리 | ["Start", "up", "code", "is", "a", "company", "."] | OOV 문제 완화. 현재 대부분의 LLM이 사용하는 방식 |
| Character Tokenization | 글자 단위 분리 | ["S", "t", "a", "r", "t", "u", "p", ...] | OOV 없음. 시퀀스가 길어져 학습 효율 저하 |

### 사용 이유

| Tokenization | 컴퓨터가 텍스트를 직접 이해할 수 없으므로, 숫자로 변환하기 위한 첫 단계로 텍스트를 일정한 단위로 나누기 위해서 |
| --- | --- |

- Tokenization 없이는 임베딩도, 모델 학습도, 추론도 시작 불가
- LLM API 과금은 **입력 토큰 수**로 결정 → 토크나이저 특성 이해로 비용 최적화 가능
- 모델의 컨텍스트 윈도우(최대 입력 토큰 수) 초과 방지에도 필수

### GPT 계열 vs BERT 계열 토크나이저

| 항목 | GPT 계열 | BERT 계열 |
| --- | --- | --- |
| 내부 방식 | BPE 또는 Byte-level BPE | WordPiece |
| 토큰 분리 특징 | 앞 공백 정보 포함 (`?is`, `?company`) | 이어지는 조각에 `##` 표기 (`play`, `##ing`) |
| 특수 토큰 | 구조 단순. 생성형에 맞는 입력 구조 | [CLS], [SEP], [PAD], [MASK] 적극 사용 |
| 주 용도 | 텍스트 생성, 대화, 요약 | 문장 분류, 감성 분석, 개체명 인식 |

> 모델과 토크나이저는 반드시 맞춰서 사용해야 함 ? 같은 문장이라도 토크나이저가 다르면 토큰 분리 결과와 정수 ID가 달라짐

### 처리 흐름

```
텍스트 → [Tokenization] → 토큰 목록 → [정수 인코딩] → 정수 ID 시퀀스 → [임베딩] → 벡터 → 모델 처리
```

### 코드 예시

```python
!pip install -q transformers torch

from transformers import AutoTokenizer

# GPT-2 토크나이저 (Hugging Face, 무료)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

sentence = "Startupcode is a company."

# 1. 토큰 분리
tokens = tokenizer.tokenize(sentence)
print(f"토큰: {tokens}")
# ['Start', 'up', 'code', '?is', '?a', '?company', '.']

# 2. 정수 인코딩
token_ids = tokenizer.encode(sentence)
print(f"정수 ID: {token_ids}")
# [10434, 929, 8189, 318, 257, 1664, 13]

# 3. 디코딩 (복원)
decoded = tokenizer.decode(token_ids)
print(f"복원: {decoded}")
# Startupcode is a company.

# 4. 토큰 수 비교 (API 비용 관련)
sentences = [
    "Startupcode is a company.",
    "Startupcode is a company that specializes in developer education.",
]
for s in sentences:
    print(f"토큰 수: {len(tokenizer.encode(s)):3d} | {s}")
```

---

## Text Embedding (텍스트 임베딩)

**텍스트를 의미를 보존한 채 고정 길이의 숫자 벡터(실수 배열)로 변환하여, 컴퓨터가 텍스트 간 유사도를 계산할 수 있게 하는 기법**

- Tokenization이 텍스트를 "조각내는" 단계라면, Text Embedding은 그 조각들을 "의미 있는 숫자로 바꾸는" 단계
- 내부적으로는 항상 Tokenization → 정수 인코딩 → 임베딩 순서로 처리됨

### 핵심 특성

| 특성 | 설명 |
| --- | --- |
| 고정 길이 | 입력 텍스트 길이와 관계없이 항상 같은 차원의 벡터 출력. 벡터 DB 저장 및 유사도 비교에 필수 |
| 의미 보존 | 비슷한 의미의 텍스트는 벡터 공간에서 가까운 위치에, 다른 의미는 먼 위치에 배치 |
| 수치 연산 가능 | 코사인 유사도, 내적, 유클리드 거리 등으로 텍스트 간 유사도를 수치로 계산 가능 |
| 모델 의존적 | 같은 텍스트라도 모델이 다르면 벡터도 다름 ? 인덱싱과 검색에 반드시 같은 모델 사용 |

### Tokenization과 Text Embedding의 관계

| 단계 | 입력 | 출력 | 핵심 역할 |
| --- | --- | --- | --- |
| Tokenization | 원본 텍스트 | 토큰 목록 | 텍스트를 처리 가능한 단위로 분리 |
| 정수 인코딩 | 토큰 목록 | 정수 ID 시퀀스 | 각 토큰에 고유 숫자 부여 |
| Text Embedding | 정수 ID 시퀀스 | 고정 길이 벡터 | 숫자에 의미를 부여하여 유사도 계산 가능하게 함 |

### 주요 임베딩 모델 비교

| 모델 | 제공자 | 차원 | 최대 입력 | 비용 | 특징 |
| --- | --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | Sentence Transformers | 384 | 256 토큰 | 무료 (로컬) | 가볍고 빠름. 학습·실험용. 영어 중심 |
| multilingual-e5-large | Microsoft | 1024 | 512 토큰 | 무료 (로컬) | 다국어 지원. 한국어 성능 우수 |
| text-embedding-3-small | OpenAI | 1536 | 8191 토큰 | API 유료 | 비용 대비 성능 우수. 긴 입력 지원 |
| text-embedding-3-large | OpenAI | 3072 | 8191 토큰 | API 유료 | 더 높은 정확도. 차원 축소 가능 |

> 임베딩 모델을 변경하면 벡터 DB 전체를 재인덱싱해야 함 ? 초기 모델 선택이 중요

### 사용 이유

| Text Embedding | 컴퓨터가 텍스트의 의미적 유사도를 계산하고, 검색·분류·군집 등 다양한 작업을 수행할 수 있게 하기 위해서 |
| --- | --- |

- RAG 파이프라인에서 **두 번** 사용: 인덱싱 시 문서 청크 벡터화 + 검색 시 질문 벡터화
- 두 단계에서 반드시 **같은 임베딩 모델** 사용해야 유사도 비교가 유효

### 코드 예시

```python
!pip install -q sentence-transformers numpy

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Adapterz is a developer book serving service provided by Startupcode.",
    "Startupcode is a company specializing in developer education.",
    "The weather is nice today.",
]

# 텍스트 → 벡터 변환
embeddings = model.encode(sentences)

print(f"벡터 배열 형태: {embeddings.shape}")  # (3, 384)
print(f"벡터 1 앞 5개: {embeddings[0][:5]}")

# 코사인 유사도 계산
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"[{i+1}] vs [{j+1}]: {sim:.4f}")
# 1번·2번(같은 주제): 높은 유사도 / 1번·3번(다른 주제): 낮은 유사도
```

---

## Dense Vector (밀집 벡터)

**임베딩 모델이 텍스트의 문맥적 의미를 모든 차원에 분산시켜 표현한 저차원 실수 벡터**

- 모든 차원에 0이 아닌 값이 채워져 있음
- 개별 숫자가 특정 의미를 나타내는 것이 아니라, 전체 벡터 패턴이 의미를 표현

### 사용 이유

| Dense Vector | 텍스트의 문맥적 의미를 벡터에 반영하여, 키워드가 일치하지 않아도 의미가 비슷한 문서를 검색할 수 있기 때문에 |
| --- | --- |

- "자동차"로 검색 시 "차량", "vehicle", "car" 관련 문서도 함께 검색 가능
- RAG 파이프라인에서 의미 기반 문서 검색의 핵심 역할

### 코드 예시

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
documents = [
    "Adapterz is a developer book serving service provided by Startupcode.",
    "Startupcode is a company specializing in developer education.",
    "RAG retrieves external data and inserts it into the LLM input.",
    "The weather is nice today.",
]

doc_vectors = model.encode(documents)

print(f"문서 수: {len(doc_vectors)}, 벡터 차원: {doc_vectors.shape[1]}")

# Dense Vector 밀집도 확인 ? 모든 차원에 값이 있음
for i, doc in enumerate(documents):
    nonzero = np.count_nonzero(doc_vectors[i])
    print(f"문서 {i+1}: {nonzero}/{len(doc_vectors[i])} 차원 (밀집도 100%)")
```

---

## Sparse Vector (희소 벡터)

**어휘 사전 크기만큼의 차원을 가지며, 텍스트에 등장한 단어 위치에만 빈도 또는 가중치 값이 있고 나머지는 모두 0인 고차원 벡터**

- 어휘 사전이 10만 개인데 문장에 등장한 단어가 15개라면, 99,985개 차원이 0

### Sparse Vector를 만드는 대표 기법

| 기법 | 값의 의미 | 특징 |
| --- | --- | --- |
| BoW (Bag of Words) | 단어 출현 횟수 | 가장 단순. 불용어("the", "is" 등)도 높은 값을 가짐 |
| TF-IDF | 단어 빈도 × 역문서 빈도 | 많은 문서에 공통으로 등장하는 단어(불용어)는 낮추고, 특정 문서의 핵심 키워드는 강조 |
| BM25 | TF-IDF + 문서 길이 정규화 | 정보 검색 표준 알고리즘. Elasticsearch, OpenSearch의 기본 알고리즘 |

> TF(Term Frequency): 해당 문서에서 단어가 얼마나 자주 등장하는지
> IDF(Inverse Document Frequency): 전체 문서 중 해당 단어가 얼마나 드물게 등장하는지
> → 두 값을 곱하면 특정 문서에서만 자주 등장하는 핵심 키워드의 가중치가 높아짐

### 사용 이유

| Sparse Vector | 정확한 키워드 매칭 기반 검색이 가능하고, 특정 단어의 출현 여부를 명확히 구분할 수 있기 때문에 |
| --- | --- |

- 제품명, 고유 명사, 에러 메시지처럼 **정확한 문자열 일치**가 중요한 검색에 적합
- 검색 결과 해석 가능 ? "이 문서가 왜 검색됐는가?"에 단어 기반으로 명확히 설명 가능

### 코드 예시

```python
import numpy as np, math

documents = [
    "Adapterz is a developer book serving service provided by Startupcode.",
    "Startupcode is a company specializing in developer education.",
    "The weather is nice today.",
]

# ===== 어휘 사전 구축 =====
def build_vocabulary(docs):
    vocab = sorted(set(w for doc in docs for w in doc.lower().replace(".", "").split()))
    return vocab, {w: i for i, w in enumerate(vocab)}

vocab, word_to_idx = build_vocabulary(documents)
print(f"어휘 사전 크기: {len(vocab)}")

# ===== TF-IDF 벡터 생성 =====
def compute_tfidf(docs, word_to_idx):
    n = len(docs)
    idf = np.zeros(len(word_to_idx))
    for word, idx in word_to_idx.items():
        df = sum(1 for doc in docs if word in doc.lower().replace(".", "").split())
        idf[idx] = math.log((n + 1) / (df + 1)) + 1

    tfidf_matrix = []
    for doc in docs:
        words = doc.lower().replace(".", "").split()
        tf = np.zeros(len(word_to_idx))
        for w in words:
            if w in word_to_idx:
                tf[word_to_idx[w]] += 1
        tf = tf / len(words) if words else tf
        tfidf_matrix.append(tf * idf)
    return np.array(tfidf_matrix)

tfidf_vectors = compute_tfidf(documents, word_to_idx)

# 희소성 확인
for i, vec in enumerate(tfidf_vectors):
    nonzero = np.count_nonzero(vec)
    total = len(vec)
    print(f"문서 {i+1}: {nonzero}/{total} 차원에 값 있음 (희소율: {(total-nonzero)/total*100:.1f}%)")
```

---

## Dense Vector vs Sparse Vector

| 항목 | Dense Vector | Sparse Vector |
| --- | --- | --- |
| 차원 | 수백~수천 (384, 768, 1536 등) | 어휘 사전 크기 (수만~수십만) |
| 값 분포 | 모든 차원에 0이 아닌 값 | 대부분 0, 소수만 0이 아님 |
| 의미 반영 | 문맥 의미 반영 | 단어 출현·빈도만 반영 |
| 강점 | 의미적 유사도 검색. 동의어 처리 | 정확한 키워드 매칭. 해석 가능 |
| 약점 | 정확한 키워드 매칭 보장 안 됨 | 동의어·유의어 처리 불가 |
| 생성 방법 | 임베딩 모델 (BERT, OpenAI 등) | BoW, TF-IDF, BM25 (통계 기반) |
| RAG 활용 | 의미 기반 문서 검색의 핵심 | 하이브리드 검색의 키워드 구성 요소 |

> "dog" 검색 시 ? Dense: "puppy", "canine" 관련 문서도 찾음 / Sparse: 정확히 "dog"이 포함된 문서만 찾음