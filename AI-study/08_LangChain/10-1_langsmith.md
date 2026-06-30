# 10-1. LangSmith 운영과 평가 (Tracing · Dataset · Evaluator · Prompt Management)

LangChain 애플리케이션을 **추적·디버깅·평가·운영**하는 통합 플랫폼

> LLM 앱은 비결정적이라 "왜 이런 답이 나왔는지" 파악이 어려움 — LangSmith가 모든 실행을 기록·분석

---

## LangSmith

LLM 앱의 실행 과정을 시각화하고 품질을 측정하는 **관측(observability)·평가 도구**

```bash
# 환경 변수만 설정하면 LangChain 코드가 자동 연동됨
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="ls-..."
export LANGSMITH_PROJECT="my-rag-project"
```

| 기능 | 역할 |
|---|---|
| Tracing | 체인 실행 단계별 입출력 추적 |
| Dataset | 평가용 입력·정답 데이터 관리 |
| Evaluator | 응답 품질 자동 채점 |
| Prompt Management | 프롬프트 버전 관리·공유 |

---

## LangSmith 실습용 RAG 베이스라인 만들기

평가의 기준이 될 **간단한 RAG 체인**을 먼저 구성

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

retriever = FAISS.from_documents(docs, OpenAIEmbeddings()).as_retriever()
prompt = ChatPromptTemplate.from_template(
    "context를 근거로 답해줘.\ncontext: {context}\n질문: {question}"
)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
)
# 환경 변수가 설정돼 있으면 invoke 시 자동으로 LangSmith에 기록됨
rag_chain.invoke("LangChain이 뭐야?")
```

> 베이스라인을 기준점으로 두고, 이후 프롬프트·검색 전략을 바꿔가며 평가 점수를 비교

---

## Tracing (추적)

체인의 **각 단계별 입력·출력·소요 시간·토큰·비용**을 트리 형태로 기록

| 추적 항목 | 내용 |
|---|---|
| 입출력 | 각 단계의 입력·출력값 |
| 지연 시간 | 단계별 실행 시간 |
| 토큰·비용 | 호출별 토큰 사용량과 비용 |
| 에러 | 실패한 단계와 예외 내용 |

> 어느 단계에서 잘못된 문서가 검색됐는지, 프롬프트가 어떻게 완성됐는지 등을 눈으로 확인 가능

---

## Dataset (데이터셋)

평가에 쓸 **입력(question)과 기대 출력(reference)** 쌍의 모음

```python
from langsmith import Client

client = Client()
dataset = client.create_dataset("rag-eval")
client.create_examples(
    inputs=[{"question": "LangChain이 뭐야?"}],
    outputs=[{"answer": "LLM 앱 개발 프레임워크"}],
    dataset_id=dataset.id,
)
```

> 프로덕션 트레이스에서 좋은/나쁜 사례를 추려 데이터셋으로 축적할 수도 있음

---

## Evaluator (평가자)

데이터셋에 대해 체인을 실행하고 응답 품질을 **자동 채점**하는 도구

| 평가 방식 | 설명 |
|---|---|
| Heuristic | 정확 일치, 길이, 포함 여부 등 규칙 기반 |
| LLM-as-Judge | LLM이 정답과 비교해 정확성·관련성 점수 부여 |
| Pairwise | 두 응답 중 더 나은 것 선택 (A/B 비교) |

```python
from langsmith.evaluation import evaluate

evaluate(
    rag_chain.invoke,            # 평가 대상
    data="rag-eval",             # 데이터셋 이름
    evaluators=[correctness_evaluator],
)
```

> RAG 평가 지표: 정확성(correctness), 충실성(faithfulness, 환각 여부), 관련성(relevance)

---

## Prompt Management (프롬프트 관리)

프롬프트를 코드와 분리해 **버전 관리·공유·재사용**하는 기능 (Prompt Hub)

```python
from langchain import hub

prompt = hub.pull("my-team/rag-prompt")   # 저장된 프롬프트 불러오기
chain = prompt | llm
```

| 기능 | 설명 |
|---|---|
| 버전 관리 | 프롬프트 변경 이력 추적·롤백 |
| 협업 | 팀이 동일 프롬프트 공유 |
| 코드 분리 | 배포 없이 프롬프트만 수정 가능 |

> 프롬프트를 바꿔도 코드 재배포가 필요 없어 빠른 실험·운영에 유리
