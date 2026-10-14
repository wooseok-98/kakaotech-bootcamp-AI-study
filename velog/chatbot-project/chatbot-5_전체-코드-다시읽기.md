<!-- 📝 글 소개 (출간 시 입력): 포트폴리오 AI 챗봇 프로젝트 5편(완결) - v1~v4 코드를 처음부터 다시 읽으며 아키텍처·RAG·에이전트를 정리하고, "왜 이렇게 짰나"를 스스로 되묻는 자체 코드 리뷰 -->

# 🎈 AI 학습 도우미 챗봇 만들기 (5) — 전체 코드를 다시 읽다

네 개의 버전(대화 → RAG → LangChain → 에이전트)을 다 만들고 나서, 문득 이런 생각이 들었다. **"이 코드를 완벽하게 내 것으로 만들어야 겠다."**

구조는 알겠는데 함수 안 코드는 시간이 지나면서 흐릿하고, "왜 이렇게 했냐"고 물으면 바로 답이 안 나왔다.
그래서 코드를 **처음부터 다시 읽으며** 정리했다.

이 글은 그 자체 코드 리뷰다. 전체 아키텍처 → 핵심 개념 → **다시 읽으며 스스로 던진 질문들** 순으로 담았다.

___

## 📌 전체 아키텍처 — 한 장으로

먼저 큰 그림. 이 프로젝트의 설계 원칙은 딱 하나다.

> **안 변하는 인프라는 `shared`에 두고, 발전하는 두뇌(`core`)만 버전별로 갈아끼운다.**

```
요청 → router(접수) → controller(흐름 제어) → core(두뇌) → LLM
                            │                    │
                            └── shared/db ───────┴── shared/embedding, 인덱스
                                (대화 저장)          (검색 인프라)

core만 교체:  v1 llm  →  v2 rag  →  v3 chains  →  v4 graph
```

| 계층 | 역할 | 버전 따라 바뀌나 |
| --- | --- | --- |
| **router** | "어느 주소 → 어느 함수" 등록 (HTTP 담당) | ❌ 거의 동일 |
| **controller** | 흐름 제어 (불러오기 → 생성 → 저장) | 🔸 조금 |
| **core** | 실제 답변 생성 (여기가 핵심) | ✅ 매번 |
| **shared** | db, embedding, 로더, 인덱스 (공용 인프라) | ❌ 공유 |

이 구조 덕분에 v1에서 v4까지 가도 **DB·라우터·컨트롤러 골격은 재사용**된다. 바뀌는 건 "답변을 어떻게 만드나"뿐이다.

___

## 📌 요청 하나의 여정

질문 하나가 들어오면 이렇게 흐른다.

```python
# router — 접수만 하는 얇은 창구
@router.post("/chat")
def chat(req: ChatRequest):
    return ChatResponse(answer=handle_chat(req.question, req.session_id))

# controller — 흐름 제어
def handle_chat(question, session_id):
    history = db.get_history(session_id)      # ① 이전 대화 불러오기
    answer  = <core 호출>                      # ② 답변 생성 (버전마다 다름)
    db.save_message(session_id, "user", question)      # ③ 저장
    db.save_message(session_id, "assistant", answer)
    return answer
```

다시 읽으며 확실히 잡은 두 가지.

- **FastAPI가 직접 부르는 건 router 함수 하나.** 그 안에서 controller를, controller가 core를 부른다. 한 단계씩 위임하는 구조다.
- **LLM은 기억이 없다(stateless).** 그래서 매번 `history`(이전 대화 전체)를 불러와 함께 보낸다. 이게 멀티턴의 원리다.

___

## 📌 v1 → v4, 무엇을 한 겹씩 쌓았나

| 버전 | core | 핵심으로 추가된 것 |
| --- | --- | --- |
| **v1** | `llm.py` | Claude API 호출 + 멀티턴 대화 |
| **v2** | `rag.py` + `llm.py` | **RAG 직접 구현** (임베딩·FAISS·검색·grounding) |
| **v3** | `chains.py` | 같은 RAG를 **LangChain LCEL 체인**으로 재구성 |
| **v4** | `tools.py` + `graph.py` | **ReAct 에이전트** (LLM이 도구를 스스로 선택) |

핵심은 **v2에서 프레임워크 없이 직접 만들고, v3에서 LangChain으로 다시 만든 것**이다. 원리를 손으로 익힌 뒤 실무 도구로 옮겼다.

___

## 📌 RAG 핵심 개념 정리

### 인덱싱(준비) vs 검색(질문) — 언제 실행되나

RAG에서 제일 헷갈렸던 지점. **두 단계는 실행 시점이 다르다.**

```
[준비 — build_index.py]  딱 한 번
   노트/위키 → 청킹 → 임베딩 → FAISS에 저장

[질문 — rag.search()]  질문마다
   질문 → 임베딩 → 유사도 비교 → 가까운 조각 top_k
```

노트 전체를 매번 임베딩하면 느리다. 그래서 **미리 한 번** 벡터로 만들어 FAISS에 저장해두고(인덱싱), 질문 때는 **질문만** 벡터로 바꿔 비교(검색)한다. 도서관에 책을 꽂아두는 것과, 사서가 찾아주는 것의 차이다.

### 임베딩 — 왜 의미 검색이 되나

글을 숫자 벡터로 바꾸는 것. 의미가 비슷하면 벡터가 가까워진다. 그래서 **"게이트가 3개인 순환 신경망?"** 이라고 물어도 **"LSTM"** 노트를 찾아낸다. 키워드 매칭이 아니라 **의미 매칭**이다.

중요한 조건: **문서와 질문을 반드시 같은 모델로 임베딩**해야 한다. 다른 모델로 벡터화하면 좌표계가 달라 비교가 무의미하다.

```python
_model = SentenceTransformer("jhgan/ko-sroberta-multitask")  # 한국어 임베딩
_model.encode(texts, normalize_embeddings=True)  # 정규화 → 내적 = 코사인 유사도
```

### FAISS — 벡터 DB지만 원문은 없다

FAISS는 **벡터만** 저장하고 유사한 걸 찾아주는 검색 엔진이다. 일반 DB(SQLite)가 "정확히 일치"를 찾는다면, FAISS는 "가까운 벡터"를 찾는다.

핵심은 **FAISS에 원문 텍스트가 없다**는 것. 벡터만 있고, "몇 번째가 가깝다"는 번호만 준다. 그 번호로 `chunks.json`에서 원문을 꺼낸다.

```python
scores, ids = index.search(qvec, top_k)   # FAISS: 점수 + 번호
for score, idx in zip(scores[0], ids[0]):
    chunk = chunks[idx]                    # 번호로 원문 꺼내기
```

### 청킹 — 왜 쪼개나

문서를 통째로 임베딩하면 의미가 뭉개지고, LLM에 넣을 때 너무 길다. 그래서 조각낸다. 노트는 `##` 헤더 단위로, 위키는 문단(~400자) 단위로 잘랐다.

### grounding — 환각을 막는 법

검색한 근거를 프롬프트에 넣고, **"이 내용만 근거로 답해, 없으면 모른다고 해"** 라고 지시한다. 이게 환각(hallucination) 억제의 핵심이다.

```python
RAG_SYSTEM = "제공된 [노트] 내용만 근거로 답해. 없으면 '노트에서 찾지 못했어요'라고 해."
```

___

## 📌 v3 — LCEL 파이프(`|`)의 데이터 흐름

v3의 유일한 새 개념. `|`는 **왼쪽 출력을 오른쪽 입력으로** 넘긴다(유닉스 파이프와 같다).

```python
rag_chain = (
    RunnablePassthrough.assign(context=lambda x: _format_docs(rag.retrieve(x["question"])))
    | prompt             # 근거+질문을 템플릿에 채움
    | llm                # Claude 호출
    | StrOutputParser()  # 응답 객체 → 텍스트
)
```

데이터가 단계마다 변신한다.

```
{question, history}
   → (검색해서 context 추가)  → {question, history, context}
   → (템플릿 채우기)          → [SystemMessage, ...history, HumanMessage]
   → (Claude 호출)          → AIMessage("LSTM은...")
   → (텍스트 추출)            → "LSTM은..."
```

v2에서 controller가 **손으로 순서대로** 하던 걸, LCEL은 **선언적으로 연결**해둔다. 실행은 `invoke` 할 때 한 번에 흐른다.

___

## 📌 v4 — 에이전트는 무엇이 다른가

### 워크플로우 vs 에이전트

| | v3 (체인) | v4 (에이전트) |
| --- | --- | --- |
| 흐름 | 내가 짠 파이프대로 **고정** | LLM이 **스스로 결정** |
| 검색 | 무조건 노트+위키 둘 다 | LLM이 어떤 도구 쓸지 고름 |
| 반복·종료 | 한 번, 파이프 끝 | LLM이 몇 번 돌지·언제 멈출지 정함 |

v3는 내가 운전하고, **v4는 LLM이 운전대를 잡는다.** 이 "LLM이 흐름을 결정"하는 게 에이전트의 정의다.

### 도구 = 함수 + docstring

```python
@tool
def search_notes(query: str) -> str:
    """사용자가 정리한 학습 노트에서 검색한다. AI/ML 개념을 물을 때 사용."""
    return rag.search_notes(query)
```

`@tool`이 함수를 도구로 만든다. 핵심은 **docstring** — LLM이 이 설명을 읽고 "이 질문엔 이 도구"라고 판단한다. 즉 docstring이 LLM용 사용설명서다.

### ReAct 루프

```python
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)   # 도구? tools : END
builder.add_edge("tools", "agent")                        # 결과 갖고 다시 agent
```

```
START → agent ──도구 원함?──▶ tools 실행 ─────┐
          │  아니면(답변 준비)                 │ (결과 갖고)
          ▼                                │
         END  ◀──────────── agent 로 복귀 ──┘
```

그래프 조립 5줄을 다시 읽으며 확실히 이해한 두 가지.

- **일반 엣지 vs 조건부 엣지**: `add_edge(A, B)`는 무조건 B로(직선). `add_conditional_edges("agent", tools_condition)`의 두 번째 인자는 **노드가 아니라 "다음 노드를 정하는 판단 함수"**다. `tools_condition`은 LLM 응답에 `tool_calls`가 있으면 `"tools"`, 없으면 `END`를 반환한다.
- **무한루프가 안 도는 이유**: `tools → agent`로 돌아와도 매번 `tools_condition` 관문을 다시 통과한다. LLM이 도구 없이 답할 준비가 되면 자연히 `END`로 빠진다. 혹시 모를 폭주는 `recursion_limit`(기본 25)로 막는다.

___

## 🔎 다시 읽으며 스스로 던진 질문들

코드를 훑으며 "이 부분, 남이 물으면 설명할 수 있나?" 싶은 곳마다 멈춰서 스스로 답해봤다. 정리해두면 나중에 나에게도, 이 글을 읽는 사람에게도 도움이 될 것 같다.

### 📌 RAG · 검색

**Q. RAG가 뭐고 왜 쓰나? 파인튜닝과 뭐가 다른가.**
검색으로 관련 문서를 찾아 그걸 근거로 답하게 하는 방식이다. 파인튜닝이 지식을 모델 안에 "암기"시킨다면, RAG는 "오픈북"처럼 외부 문서를 참고한다. 지식이 자주 바뀌거나 출처가 필요할 때 RAG가 유리하고, 문서를 갈아끼우면 바로 반영된다.

**Q. 임베딩이 어떻게 의미 검색을 가능하게 하나.**
글을 의미 벡터로 바꾸면, 의미가 비슷한 글끼리 벡터가 가까워진다. 그래서 표현이 달라도(예: "게이트 3개 순환 신경망" ↔ "LSTM") 찾아낸다. 키워드가 아니라 의미로 매칭된다.

**Q. 질문도 문서와 같은 모델로 임베딩하는 이유는.**
같은 의미 공간에 있어야 거리가 의미를 가진다. 다른 모델로 벡터화하면 좌표계가 달라 유사도 비교가 무의미해진다.

**Q. FAISS는 DB인가. 왜 FAISS를 썼나.**
벡터를 저장하고 유사한 걸 찾아주는 벡터 검색 라이브러리다. 로컬에서 가볍고 빠르며 무료라, 수백~수천 조각 규모엔 충분해서 골랐다. 대규모 서비스라면 Pinecone 같은 관리형을 고려할 것 같다.

**Q. 벡터 DB에 원문 텍스트가 저장되나.**
FAISS에는 벡터만 있고 원문은 없다. 검색하면 "몇 번째가 가깝다"는 번호를 주고, 그 번호로 별도의 `chunks.json`에서 원문을 꺼낸다.

**Q. 유사도는 어떻게 계산하나.**
코사인 유사도다. 임베딩을 정규화했기 때문에 내적이 곧 코사인이 되고, FAISS `IndexFlatIP`(내적)로 값이 큰 순서대로 가져온다.

**Q. 청킹은 어떻게, 왜 했나.**
노트는 `##` 헤더 단위, 위키는 ~400자 문단 단위로 잘랐다. 통째로 임베딩하면 의미가 뭉개지고 LLM 입력도 길어져서, 검색·주입에 적당한 크기로 쪼갰다.

**Q. 환각(hallucination)은 어떻게 막았나.**
grounding 프롬프트로 "제공된 근거만 사용하고, 없으면 모른다고 하라"고 지시했다. 그리고 답변에 출처를 밝히게 했다.

### 📌 아키텍처 · 멀티턴

**Q. 멀티턴 대화는 어떻게 구현했나. LLM은 기억이 없는데.**
LLM은 stateless라서, 매 요청마다 SQLite에서 이전 대화를 불러와 이번 질문과 함께 통째로 보낸다. `session_id`로 대화를 구분한다.

**Q. 사용자를 어떻게 구분하나.**
`session_id`로 구분한다. 프론트에서 localStorage에 고정된 id를 두고 매 요청에 실어 보내, 재방문해도 대화가 이어진다.

**Q. 버전을 어떻게 나눴나.**
안 변하는 인프라(DB·임베딩·인덱스)는 `shared`로 공유하고, 발전하는 두뇌(`core`)만 버전별로 교체했다. 그래서 router·controller·DB 골격은 v1~v4가 재사용한다. 관심사 분리와 DRY를 실제로 적용한 부분이다.

### 📌 LangChain · 에이전트

**Q. v2에서 직접 만든 RAG를 왜 v3에서 LangChain으로 다시 짰나.**
직접 만들면 원리는 알지만 코드가 흩어지고 조합이 어렵다. LCEL은 검색→프롬프트→LLM→파싱을 `|`로 선언적으로 조합해 가독성과 확장성이 좋다. **v2로 원리를 익히고, v3로 실무 방식을 익힌** 셈이다.

**Q. 워크플로우와 에이전트의 차이는.**
흐름을 개발자가 정하면 워크플로우, LLM이 런타임에 정하면 에이전트다. v3는 고정 파이프고, v4는 LLM이 어떤 도구를 쓸지·몇 번 쓸지·언제 멈출지 스스로 결정한다.

**Q. LLM이 도구를 어떻게 고르나.**
각 도구를 `@tool`로 등록하고, docstring에 용도를 적는다. LLM은 이 설명을 읽고 판단한다. `bind_tools`로 도구 목록을 쥐여주면, LLM이 답변 대신 "이 도구를 이 인자로 불러라"는 구조화된 호출 지시(`tool_calls`)를 만들 수 있다.

**Q. 에이전트가 무한루프에 빠지지 않나.**
정상적으로는 LLM이 도구 없이 답할 준비가 되면 `tools_condition`이 `END`로 보내 종료된다. 혹시 LLM이 도구 호출을 멈추지 못하는 경우를 대비해 `recursion_limit`(기본 25)로 강제 중단 안전장치를 둔다.

### 📌 한계 · 개선점 (솔직하게)

**Q. 이 프로젝트의 한계나 개선점은.**
세 가지다. ① **정량 평가 미완** — RAGAS로 faithfulness·answer relevancy를 측정하는 게 다음 단계다. ② **청킹이 단순** — 고정 크기라 문맥이 잘리는 경우가 있어 겹침(overlap)이나 구조 기반 청킹을 개선하고 싶다. ③ **검색 품질** — 지금은 단순 top_k라, reranking이나 하이브리드 검색을 붙이면 정확도가 오를 것 같다.

___

## 🎈 다시 읽고 배운 점

- **구조를 아는 것과 코드를 아는 것은 다르다.** 파일 역할은 알았지만, 함수 안에서 데이터가 어떻게 변하는지는 다시 읽고서야 잡혔다.
- **모든 설계엔 "왜"가 있다.** shared/core 분리, grounding, recursion_limit 하나하나가 이유가 있었고, 그 이유를 말할 수 있어야 내 프로젝트다.
- **모르는 건 모른다고.** RAGAS 평가는 아직 안 했다. 안 한 건 안 했다고 적고, "다음 단계로 이렇게 하겠다"고 두는 게 낫다.

v1부터 v4까지 한 겹씩 쌓고, 다섯 번째 글에서 처음으로 돌아가 다시 읽었다. 이제야 이 코드가 **내 것** 같다.

> 코드 전체: [github.com/wooseok-98/AI-study-chatbot](https://github.com/wooseok-98/AI-study-chatbot)
