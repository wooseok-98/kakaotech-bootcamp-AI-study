# 1-1. LangChain 기초 (LangChain · Chat Model · Messages · LCEL)

LLM 애플리케이션 개발에 필요한 프롬프트·모델·출력 파서·검색·메모리 등의 컴포넌트를 **표준화된 인터페이스**로 제공하는 오픈소스 프레임워크

> 핵심 아이디어: LLM 앱의 각 단계(프롬프트 작성 → 모델 호출 → 출력 파싱 → 외부 데이터 검색)를 독립 컴포넌트로 분리하고 파이프라인으로 연결

---

## LangChain

RAG를 직접 구현하면 임베딩·검색·프롬프트·생성 함수를 각각 만들어 순서대로 호출 — LangChain은 이 각 단계를 **표준 컴포넌트로 제공**하고 파이프(`|`) 연산자로 연결

```bash
pip install langchain langchain-openai
```

### 핵심 모듈

기본 구성요소(Model/Messages, Retrieval, Memory, Tools)를 조합해 **Chains(LCEL)** 또는 **Agents**를 구성

| 모듈 | 역할 | 주요 컴포넌트 |
|---|---|---|
| **Model / Messages** | LLM 입출력 관리 — 프롬프트 구성·모델 호출·메시지 형식 대화 관리 | ChatPromptTemplate, ChatModel, StrOutputParser, PydanticOutputParser |
| **Retrieval / Indexing** | 외부 데이터 로딩·분할·저장·검색 (RAG) | DocumentLoader, TextSplitter, VectorStore, Retriever |
| **Chains (LCEL)** | 컴포넌트를 파이프(`|`)로 선언적으로 연결 | RunnableSequence, RunnablePassthrough, RunnableParallel |
| **Short-term Memory** | 대화 기록 저장·다음 호출에 주입 | RunnableWithMessageHistory, ChatMessageHistory |
| **Tools** | 모델이 외부 기능을 호출하도록 정의 | Tool, `@tool` |
| **Agents** | 모델이 어떤 도구를 쓸지 판단해 작업 수행하는 고수준 구조 | create_agent |

> **Chains vs Agents**: LCEL은 컴포넌트를 **연결하는 방식**, Agents는 모델이 도구를 **스스로 선택 실행**하는 구조 — 상태 기반 분기·반복·복잡한 오케스트레이션은 LangGraph에서 다룸

### RAG 직접 구현과의 대응

| 직접 구현 단계 | LangChain 컴포넌트 |
|---|---|
| Document Loading | `DocumentLoader` |
| Chunking | `TextSplitter` |
| 벡터 저장 | `VectorStore` |
| Retrieval | `Retriever` |
| Generation | `ChatModel` + `StrOutputParser` |
| 대화 기록 관리 | `RunnableWithMessageHistory` / `ChatMessageHistory` |

> 표준 인터페이스 덕분에 검색기·모델·프롬프트·출력 처리를 **쉽게 교체하면서도 같은 구조 유지**

### 사용 이유

하드코딩은 프로토타입엔 빠르지만, 프로젝트가 커지면 모델 교체·프롬프트 관리·실행 추적을 모두 직접 구현해야 함

| 문제 (하드코딩) | LangChain 해결 |
|---|---|
| 모델 교체 시 호출·파싱 코드 전면 수정 | `ChatOpenAI()` → `ChatAnthropic()` 객체만 교체 |
| f-string으로 프롬프트 직접 조합 | `ChatPromptTemplate`로 변수·구조 체계적 관리 |
| RAG에 단계 추가 시 흐름 분석·삽입 | 파이프 중간에 컴포넌트 끼워 넣기 |
| 실행 로그 수동 기록 | LangSmith 연동으로 각 단계 자동 추적 |

### 사용 흐름

```
pip install → 컴포넌트 import → LCEL 파이프(|)로 체인 구성 → invoke/stream/batch로 실행
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {topic} 전문가입니다. 간결하게 답변하세요."),
    ("human", "{question}"),
])
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()          # 응답에서 텍스트만 추출

chain = prompt | model | parser     # LCEL 체인
print(chain.invoke({"topic": "번역", "question": "'안녕'을 영어로"}))
```

> 한 번 구성한 체인은 입력만 바꿔 재사용 — 매번 f-string으로 프롬프트를 다시 조합할 필요 없음

### 패키지 구조

역할별로 나뉜 여러 패키지를 조합 — 모델 제공자별 의존성 분리, 필요한 기능만 설치

| 패키지 | 역할 |
|---|---|
| `langchain-core` | Runnable·PromptTemplate·Messages·OutputParser·LCEL 등 기본 인터페이스 |
| `langchain` | 프레임워크 본체 (v1 기준 핵심 빌딩 블록 중심으로 경량화) |
| `langchain-openai` | OpenAI 모델·임베딩 통합 (`ChatOpenAI`) |
| `langchain-google-genai` | Gemini 통합 (`ChatGoogleGenerativeAI`) |
| `langchain-community` | 커뮤니티 관리 서드파티 통합 (벡터스토어, 로더 등) |
| `langchain-text-splitters` | 문서 분할 전용 (`RecursiveCharacterTextSplitter`) |

### 생태계

개발 → 디버깅 → 평가 → 배포로 이어지는 도구 모음

| 도구 | 역할 |
|---|---|
| **LangSmith** | 체인 실행 추적·디버깅·프롬프트 테스트·평가·모니터링 |
| **LangGraph** | 상태 기반 워크플로우·분기·반복·Human-in-the-Loop 등 고급 실행 구조 |
| **LangServe** | LangChain 체인을 REST API로 배포 |

---

## Chat Model

여러 LLM 제공자를 **같은 방식으로 호출**할 수 있게 하는 표준 인터페이스 — 특정 모델 하나가 아님

> 문자열 하나보다 `SystemMessage`·`HumanMessage`·`AIMessage` 같은 **역할별 메시지 기반 호출**이 기본

### 제공자별 클래스

클래스·패키지는 달라도 모두 동일한 인터페이스(`invoke`, `stream`, `batch`)를 공유 → 모델만 교체하면 됨

| 클래스 | 제공자 | 패키지 | 대표 모델 |
|---|---|---|---|
| `ChatOpenAI` | OpenAI | langchain-openai | gpt-4o-mini |
| `ChatAnthropic` | Anthropic | langchain-anthropic | claude-sonnet-4 |
| `ChatGoogleGenerativeAI` | Google | langchain-google-genai | gemini-2.5-flash |
| `ChatOllama` | Ollama (로컬) | langchain-ollama | llama3, gemma2 |

### 주요 파라미터

| 파라미터 | 설명 | 비고 |
|---|---|---|
| `model` | 사용할 모델명 | 필수 |
| `temperature` | 출력 무작위성 — 0에 가까울수록 결정적, 높을수록 다양 | 번역·요약은 낮게, 창작은 높게 |
| `max_tokens` | 생성할 최대 출력 길이 | 초과 시 응답 중간 잘림 |
| `api_key` | 인증 키 | 코드보다 환경 변수 관리 권장 |

### AIMessage 객체 구조

`model.invoke()`가 반환하는 `AIMessage`에는 텍스트 외 메타데이터 포함

```python
response = model.invoke(messages)
print(response.content)            # 실제 답변 텍스트
print(response.response_metadata)  # 모델명, 종료 이유 등
print(response.usage_metadata)     # 입력·출력·총 토큰 수
```

> `usage_metadata`는 비용 추적, `response_metadata`는 응답이 왜 잘렸는지 확인할 때 유용

---

## Messages

Chat Model의 입출력을 구성하는 **역할별 메시지 객체의 집합** — `langchain_core.messages`에서 import

> 문자열 하나로 이어 붙이면 규칙·질문·이전 답변이 한 덩어리로 섞임 — 역할을 나눠야 모델이 맥락을 정확히 이해

### 메시지 타입

| 클래스 | 역할 | 설명 |
|---|---|---|
| `SystemMessage` | system | 모델의 행동 방식·페르소나·규칙 지정 (대화 전체에 일관 적용) |
| `HumanMessage` | human | 사용자의 질문·요청 |
| `AIMessage` | ai | 모델의 응답 (대화 히스토리 구성 시에도 사용) |
| `ToolMessage` | tool | 도구 실행 결과를 모델에 전달 (에이전트 구조) |

### BaseMessage 공통 속성

모든 메시지 클래스는 역할은 달라도 `BaseMessage` 기반의 공통 구조 공유

| 속성 | 설명 |
|---|---|
| `content` | 본문 텍스트 (문자열 또는 멀티모달 콘텐츠 블록 리스트) |
| `type` | 역할 문자열 (`"system"`, `"human"`, `"ai"`, `"tool"`) |
| `response_metadata` | 모델명·완료 이유·토큰 사용량 등 (주로 AIMessage) |
| `additional_kwargs` | 제공자별 추가 정보 (예: OpenAI function_call) |
| `id` / `name` | 메시지 식별자 / 발신자 이름 (선택) |

### 기본 사용

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="문서를 참고해 답하고, 없으면 모른다고 답하세요."),
    HumanMessage(content="어댑터즈의 교수법은?"),
]
response = model.invoke(messages)   # AIMessage 반환 (type == "ai")
```

### 멀티턴 대화 히스토리

이전 대화를 `AIMessage`로 포함하면 모델이 맥락 이해 (리스트에 순서대로 쌓기)

```python
messages = [
    SystemMessage(content="문서를 참고해 답하세요."),
    HumanMessage(content="어댑터즈의 교수법은?"),
    AIMessage(content="5단 분석법입니다."),        # 이전 응답
    HumanMessage(content="그 교수법의 단계를 설명해줘"),  # "그 교수법" = 5단 분석법
]
```

> 메시지가 Python 리스트이므로 슬라이싱·필터링으로 조작 가능 — 최근 N턴만 유지: `messages[-2*N:]`, 역할별 필터: `[m for m in messages if m.type == "human"]`

```python
# 대화 트리밍: SystemMessage는 유지, 최근 대화만 남김
system = [m for m in all_messages if m.type == "system"]
recent = [m for m in all_messages if m.type != "system"][-4:]   # 최근 2턴
trimmed = system + recent
```

> 토큰 한도 관리를 위한 자동 트리밍(`trim_messages`)은 [[5-1_message-history]]에서 다룸

> **순서 주의**: Chat Model은 메시지를 위→아래로 읽음 — `SystemMessage`를 맨 앞에 두고 Human/AI를 대화 순서대로 배치. AI가 Human보다 앞서면 흐름을 잘못 이해

---

## LCEL (LangChain Expression Language)

파이프(`|`) 연산자로 **Runnable 컴포넌트를 선언적으로 연결**해 체인을 구성하는 핵심 문법

> 왼쪽 컴포넌트의 출력이 오른쪽의 입력으로 자동 전달 (유닉스 파이프와 동일한 개념)

### 하드코딩 vs LCEL

```python
# 하드코딩 — 각 단계 직접 호출
prompt_result = prompt.invoke({"question": "어댑터즈란?"})
model_result = model.invoke(prompt_result)
parser_result = parser.invoke(model_result)

# LCEL — 한 줄로 연결
chain = prompt | model | parser
result = chain.invoke({"question": "어댑터즈란?"})
```

> 코드 줄 수보다, 체인 구조가 `prompt | model | parser` **한 줄에 명확히 드러나는 것**이 핵심 — 파악·수정·확장이 쉬움 (검색 추가 시 `retriever | prompt | model | parser`)

### Runnable 인터페이스

파이프로 연결 가능한 이유는 주요 컴포넌트가 모두 **Runnable 인터페이스**를 구현하기 때문

| 메서드 | 설명 |
|---|---|
| `invoke(input)` | 단일 입력 실행 |
| `stream(input)` | 결과를 청크(토큰) 단위로 스트리밍 |
| `batch([...])` | 여러 입력 병렬 처리 |
| `ainvoke / astream / abatch` | 각각의 비동기 버전 |

> LCEL 체인은 별도 구현 없이 스트리밍·배치를 자동 지원 — 상세 구조는 [[4-1_lcel-runnables]]에서 다룸

```python
for chunk in chain.stream({"question": "..."}):
    print(chunk, end="", flush=True)   # 토큰 단위 실시간 출력
```

### LCEL vs 함수 직접 호출

| 방식 | 특징 | 적합한 상황 |
|---|---|---|
| **LCEL** | 파이프로 선언적 구성, 구조가 한 줄에 드러남, stream/batch 자동 지원 | 여러 컴포넌트를 조합하는 LLM 앱 전반 |
| 함수 직접 호출 | 각 단계 명시적 호출, 프레임워크 의존성 없음, stream/batch 직접 구현 | 단순 단일 호출, 학습, 세부 제어 필요 시 |

---

## 참고: LangChain 버전 변화 흐름

| 시기 | 핵심 변화 |
|---|---|
| 초기 | `LLMChain`·`SequentialChain` 등 **클래스 기반 체인** 중심 — 간단하나 구조가 경직 |
| v0.1 (2024.01) | `langchain-core`·`langchain-community`로 패키지 재정리, **LCEL 조합 방식**이 핵심으로 정착 |
| v1.0 (2025.10) | 본체 경량화, **에이전트 중심 구조**로 재구성 — 복잡한 상태 흐름은 LangGraph가 담당 |

> 예전 예제에 `LLMChain`이 많고 지금은 `prompt | model | parser`를 쓰는 이유 — 클래스형 체인에서 LCEL 조합으로, 다시 에이전트 중심으로 진화
