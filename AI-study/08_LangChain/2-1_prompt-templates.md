# 2-1. 프롬프트 구성 (PromptTemplate · ChatPromptTemplate · FewShotPromptTemplate · MessagesPlaceholder)

변수 자리표시자(`{variable}`)를 포함한 프롬프트를 **객체로 관리**하고, 실행 시점에 값을 주입해 완성된 프롬프트를 생성하는 템플릿들

> 프롬프트를 f-string으로 매번 조합하지 않고, 틀은 고정하고 값만 바꿔 반복 사용 — 재사용성·일관성·버전 관리 확보

---

## PromptTemplate

변수 자리표시자를 포함한 **단일 문자열 프롬프트** 템플릿

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("{topic}에 대해 {length}자 이내로 설명하세요")
print(prompt.input_variables)   # ['length', 'topic'] — 변수 자동 감지
print(prompt.template)          # 원본 템플릿 문자열
```

> `from_template()`은 문자열을 그대로 저장하지 않고 **원본 템플릿 + 필요한 변수 목록 + 값 주입 기능**을 가진 객체를 반환

### 생성 방법

| 방법 | 설명 |
|---|---|
| `PromptTemplate.from_template(str)` | 문자열에서 `{변수}` **자동 감지** (권장, 간결) |
| `PromptTemplate(template=..., input_variables=[...])` | 변수 목록을 **직접 명시** |

```python
# 직접 생성 — input_variables 명시
prompt = PromptTemplate(
    template="[문서]\n{document}\n\n[질문]\n{question}",
    input_variables=["document", "question"],
)
```

### format() vs invoke()

둘 다 값을 주입해 완성된 프롬프트를 만들지만 **반환 타입이 다름**

| 메서드 | 인자 형태 | 반환 타입 | 적합한 상황 |
|---|---|---|---|
| `format(변수="값")` | 키워드 인자 | `str` (문자열) | 결과 확인·디버깅, 문자열로 바로 사용 |
| `invoke({"변수": "값"})` | 딕셔너리 | `StringPromptValue` | LCEL 체인에서 다음 컴포넌트로 전달 |

```python
prompt.format(topic="어댑터즈", length="100")     # "어댑터즈에 대해 100자..." (str)
prompt.invoke({"topic": "RAG", "length": "200"})  # StringPromptValue (.text로 접근)
```

### partial() — 일부 변수 미리 고정

항상 같은 값이 들어가는 변수를 **미리 채운 새 템플릿**을 반환 (원본은 유지)

```python
prompt = PromptTemplate.from_template("당신은 {role} 전문가입니다. {topic}을 설명하세요")
partial_prompt = prompt.partial(role="교육")     # role 고정
partial_prompt.format(topic="5단 분석법")         # topic만 주입
```

> 역할·출력 언어·형식처럼 반복되는 고정값을 미리 채워 두면, 매번 바뀌는 변수만 주입 — 일관성 유지 + 실수 방지

### vs f-string

| 방식 | 특징 | 적합한 상황 |
|---|---|---|
| **PromptTemplate** | `input_variables`로 변수 관리, `partial()` 지원, LCEL 파이프 연결, 객체 분리로 버전 관리 | 체인 구성·재사용·버전 관리 |
| f-string 직접 조합 | Python 기본 문법, 의존성 없음, 변수 관리 기능 없음 | 빠른 테스트, 단순 단일 프롬프트 |

---

## ChatPromptTemplate

역할별(System/Human/AI) **메시지 리스트**를 구성하는 프롬프트 템플릿 — Chat Model의 표준 입력 형식

> PromptTemplate은 단일 문자열, ChatPromptTemplate은 **역할이 구분된 메시지 리스트**를 생성 — 시스템 지시와 사용자 질문을 분리한 상태로 변수 주입

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role} 전문가입니다"),
    ("human", "{question}"),
])
messages = prompt.invoke({"role": "번역", "question": "'안녕'을 영어로"})
```

### 튜플 키워드 ↔ 메시지 타입

| 튜플 키워드 | 생성 메시지 | 역할 |
|---|---|---|
| `"system"` | SystemMessage | 모델의 행동 규칙 |
| `"human"` / `"user"` | HumanMessage | 사용자 질문·요청 |
| `"ai"` / `"assistant"` | AIMessage | 모델의 이전 응답 (히스토리용) |

> `("system", "...")` 튜플 방식은 내부적으로 해당 메시지 객체로 변환 — `SystemMessagePromptTemplate.from_template(...)` 메시지 클래스 방식보다 간결해 실무에서 주로 사용

### format_messages() vs invoke()

| 메서드 | 반환 타입 | 적합한 상황 |
|---|---|---|
| `format_messages(...)` | 메시지 객체 **리스트** (`[SystemMessage, HumanMessage]`) | 메시지 내용 직접 확인·수정 |
| `invoke({...})` | `ChatPromptValue` (`.messages`로 리스트 접근) | LCEL 체인에서 다음 컴포넌트로 전달 |

> PromptTemplate의 `format()`/`invoke()` 관계와 동일한 구조 — 문자열 기반이냐 메시지 기반이냐만 다름

### vs PromptTemplate

| 클래스 | 생성물 | 비고 |
|---|---|---|
| **ChatPromptTemplate** | 역할별 메시지 리스트 | Chat Model 표준, 시스템/질문 분리, 실무 주력 |
| PromptTemplate | 단일 문자열 | Chat Model 전달 시 `HumanMessage` 하나로 변환 → **SystemMessage 분리 불가** |

> `partial()`은 ChatPromptTemplate에서도 동일하게 사용 가능

---

## FewShotPromptTemplate

소수(2~5개)의 **입출력 예시를 프롬프트에 자동 삽입**해, 모델이 원하는 형식·패턴으로 응답하도록 유도

> "이렇게 하라"고 설명하는 대신 "이런 식으로 하라"고 **예시로 보여주는** 방식 — 출력 형식 통일, 톤 조절, 분류 기준 전달에 유용

### Few-Shot vs Zero-Shot

| 방식 | 설명 | 적합한 상황 |
|---|---|---|
| **Zero-Shot** | 예시 없이 지시만 전달 | 일반적 작업, 모델이 잘 아는 것, 빠른 테스트 |
| **Few-Shot** | 2~5개 입출력 예시 제공 → 패턴 모방 | 출력 형식 통일, 도메인 특화 분류, 톤 조절 |

> 예시는 많다고 좋은 게 아님 — 2~5개면 충분, 개수보다 **품질**(원하는 패턴을 명확히 보여주는 대표 예시)이 중요. 많으면 Context Window 낭비

### 구성 요소

최종 프롬프트 = **`prefix` + 예시들 + `suffix`** 순서로 조합

| 구성 요소 | 설명 |
|---|---|
| `examples` | 입출력 예시 딕셔너리 리스트 (각 딕셔너리 = 예시 1건 = 1 shot) |
| `example_prompt` | 예시 1건을 어떤 형식으로 표현할지 정의하는 PromptTemplate |
| `prefix` | 예시들 앞에 붙는 작업 설명·지시 |
| `suffix` | 예시들 뒤에 붙는 실제 입력 자리 (`{변수}` 포함) |

```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"term": "RAG", "definition": "외부 문서를 검색해 LLM 입력에 삽입 후 답변 생성하는 구조"},
    {"term": "임베딩", "definition": "텍스트를 고정 길이 숫자 벡터로 변환하는 방법"},
]
example_prompt = PromptTemplate(
    input_variables=["term", "definition"],
    template="용어: {term}\n정의: {definition}",
)
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="다음 예시를 참고해 같은 형식으로 정의하세요",
    suffix="용어: {term}\n정의:",       # 모델이 채울 자리
    input_variables=["term"],
)
print(prompt.format(term="LCEL"))
```

### 단일 문자열 방식의 한계 → FewShotChatMessagePromptTemplate

`FewShotPromptTemplate`은 예시를 **하나의 문자열 블록**으로 이어 붙임 → 모델이 `suffix`의 `용어:`/`정의:` 접두어까지 따라 써서 **출력에 중복** 발생 가능

> 해결: 예시를 **Human → AI 메시지 쌍**으로 구성하면 역할 구조가 입력/출력 경계를 명확히 해 접두어 없이도 패턴 유지

```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

chat_examples = [
    {"input": "RAG", "output": "외부 문서를 검색해 LLM 입력에 삽입 후 답변 생성하는 구조"},
    {"input": "임베딩", "output": "텍스트를 고정 길이 숫자 벡터로 변환하는 방법"},
]
# 접두어("용어:") 없이 값만 — 역할 구조가 패턴을 전달하므로 접두어 불필요
example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
few_shot = FewShotChatMessagePromptTemplate(examples=chat_examples, example_prompt=example_prompt)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 예시를 참고해 기술 용어를 한 줄로 정의하세요"),
    few_shot,                    # 예시들이 Human/AI 쌍으로 삽입
    ("human", "{input}"),
])
```

| 구분 | FewShotPromptTemplate | FewShotChatMessagePromptTemplate |
|---|---|---|
| 예시 형식 | 단일 문자열 블록 | Human → AI 메시지 쌍 |
| 접두어 | 필요 (입출력 경계 구분용) | 불필요 (역할 구조가 경계 제공) |
| 궁합 | 일반 LLM | Chat Model (출력 더 깔끔) |

### ExampleSelector — 예시 동적 선택

전체 예시 중 **현재 입력에 맞는 예시만 골라** 삽입 → Context Window 절약

| 방식 | 선택 기준 | 적합한 상황 |
|---|---|---|
| `LengthBasedExampleSelector` | 길이 기준 개수 제한 | 토큰 한도 관리 |
| `SemanticSimilarityExampleSelector` | 입력과 의미적으로 유사한 예시 | 관련 예시 선별 |
| `MaxMarginalRelevanceExampleSelector` | 관련성 + 다양성 | 예시 많고 중복 심할 때 |

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

selector = SemanticSimilarityExampleSelector.from_examples(
    examples, OpenAIEmbeddings(), Chroma, k=2,   # 입력과 가장 유사한 2개만 선택
)
```

---

## MessagesPlaceholder

`ChatPromptTemplate` 안에서 실행 시점에 **메시지 리스트가 삽입될 위치를 지정**하는 컴포넌트

> 대화 기록처럼 매 요청마다 달라지는 메시지 흐름을 **역할 구조(System/Human/AI) 그대로** 주입 — 멀티턴 대화 맥락 유지

### {variable} vs MessagesPlaceholder

| 삽입 방식 | 삽입되는 값 |
|---|---|
| `{variable}` | 단일 문자열 → 하나의 메시지 안에 텍스트로 삽입 (`("human", "{question}")` → HumanMessage 1개) |
| `MessagesPlaceholder` | 메시지 객체 **리스트** → 여러 메시지가 역할 구조 유지한 채 삽입 (대화 히스토리 등) |

> `{variable}`은 메시지 1개 안의 텍스트, MessagesPlaceholder는 메시지 리스트 전체 — 역할 구조가 유지되어 모델이 이전 발화가 질문인지 응답인지 구분

### 사용 방법

`from_messages()`에 `MessagesPlaceholder(variable_name="history")` 배치 → `invoke()` 시 해당 변수에 **메시지 리스트** 전달

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 문서를 참고해 답하세요\n\n{document}"),
    MessagesPlaceholder(variable_name="history"),   # 대화 기록이 삽입될 자리
    ("human", "{question}"),
])
print(prompt.input_variables)   # ['document', 'history', 'question']

chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
```

```python
# 첫 질문 — 히스토리 없음 (빈 리스트)
chain.invoke({"document": doc, "history": [], "question": "어댑터즈는 무엇인가요?"})

# 후속 질문 — 이전 대화를 history에 포함
history = [
    HumanMessage(content="어댑터즈는 무엇인가요?"),
    AIMessage(content="스타트업코드의 개발 교재 서빙 서비스입니다"),
]
chain.invoke({"document": doc, "history": history, "question": "거기서 쓰는 교수법은?"})
# 모델이 "거기서" = 어댑터즈로 연결해 해석
```

> 히스토리가 없는 첫 대화와 여러 턴이 쌓인 대화를 **같은 템플릿 구조로** 처리 — 실행 시 history 리스트만 바꿔 재사용

### optional 파라미터

기본적으로 `history`는 필수 변수 — 값을 안 넣으면 오류. `optional=True`로 선언하면 **없어도 빈 리스트로 처리**

```python
MessagesPlaceholder(variable_name="history", optional=True)
# → history 생략해도 정상 동작, 첫 질문에서 history=[] 반복 전달 불필요
```

> `RunnableWithMessageHistory`와 결합하면 대화 기록을 이 자리에 **자동 주입** — [[5-1_message-history]]에서 다룸
