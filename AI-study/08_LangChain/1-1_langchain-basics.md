# 1-1. LangChain 기초 (LangChain · Messages · LCEL)

LLM, 프롬프트, 데이터, 외부 도구를 **체인(Chain)** 으로 연결해 AI 애플리케이션을 만드는 프레임워크

> LLM 호출 한 번으로 끝나지 않는 작업(검색→요약→응답 등)을 모듈로 쪼개 조립할 수 있게 해줌

---

## LangChain

### 핵심 패키지 구조

| 패키지 | 역할 |
|---|---|
| `langchain-core` | Runnable, 프롬프트, 메시지, 파서 등 핵심 추상화 |
| `langchain` | 체인, 에이전트 등 상위 조합 로직 |
| `langchain-community` | 서드파티 통합 (로더, 벡터스토어 등) |
| `langchain-openai` | OpenAI 모델·임베딩 전용 통합 |
| `langchain-text-splitters` | 텍스트 분할 도구 |

```bash
pip install langchain langchain-openai langchain-community
```

### 주요 구성 요소

| 구성 요소 | 설명 |
|---|---|
| **Model** | LLM / ChatModel — 입력을 받아 텍스트·메시지 생성 |
| **Prompt** | 모델에 전달할 입력을 템플릿화 |
| **Output Parser** | 모델 출력을 원하는 형식(문자열, JSON, 객체)으로 변환 |
| **Chain** | 위 요소들을 `|` 로 연결한 실행 파이프라인 |
| **Retriever** | 외부 데이터에서 관련 문서를 검색 (RAG의 핵심) |

### 기본 호출

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke("LangChain을 한 문장으로 설명해줘")
print(response.content)   # AIMessage의 텍스트
```

> `temperature`: 0에 가까울수록 일관된 답변, 높을수록 다양·창의적 답변

---

## Messages

대화형 ChatModel이 주고받는 **역할(role)을 가진 메시지 객체**

| 메시지 | 역할 |
|---|---|
| `SystemMessage` | 모델의 행동·페르소나 지시 (대화 맨 앞) |
| `HumanMessage` | 사용자의 입력 |
| `AIMessage` | 모델의 응답 |
| `ToolMessage` | 도구(함수) 호출 결과 |

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="너는 친절한 한국어 비서야."),
    HumanMessage(content="파이썬이 뭐야?"),
]
response = llm.invoke(messages)
print(response.content)
```

> 문자열 하나만 넘기면 내부적으로 `HumanMessage` 하나로 변환됨

---

## LCEL (LangChain Expression Language)

파이프 연산자 `|` 로 컴포넌트를 **선언적으로 연결**하는 LangChain의 체인 문법

> 앞 단계의 출력이 다음 단계의 입력으로 흐름 (유닉스 파이프와 동일한 개념)

### 기본 체인

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("{topic}에 대해 한 문장으로 설명해줘")
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()        # AIMessage → 문자열

chain = prompt | llm | parser     # LCEL 체인
print(chain.invoke({"topic": "RAG"}))
```

흐름: `{"topic": "RAG"}` → 프롬프트 완성 → LLM 호출 → 텍스트만 추출

### LCEL의 장점

| 장점 | 설명 |
|---|---|
| 일관된 인터페이스 | 모든 컴포넌트가 `invoke` / `batch` / `stream`을 동일하게 지원 |
| 스트리밍 | `chain.stream()`으로 토큰 단위 출력 자동 지원 |
| 비동기 | `chain.ainvoke()`로 비동기 실행 |
| 병렬·분기 | Runnable 조합으로 복잡한 흐름 구성 가능 |

### 실행 메서드

| 메서드 | 설명 |
|---|---|
| `invoke(input)` | 입력 1개 실행 |
| `batch([...])` | 여러 입력 병렬 실행 |
| `stream(input)` | 결과를 청크 단위로 스트리밍 |
| `ainvoke / abatch / astream` | 각각의 비동기 버전 |

```python
for chunk in chain.stream({"topic": "벡터 임베딩"}):
    print(chunk, end="", flush=True)   # 토큰 단위 실시간 출력
```
