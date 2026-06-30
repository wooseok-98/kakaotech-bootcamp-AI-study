# 3-1. LangGraph 핵심 (Tool Calling · 제어 흐름 · ReAct)

LLM이 **도구를 호출**하고, 그래프가 **분기·병렬·반복**하며 에이전트를 구성하는 핵심 메커니즘

> 도구 호출 + 제어 흐름(라우팅·루프)을 조합하면 ReAct 같은 자율 에이전트가 만들어짐

---

## Tool Calling

LLM이 도구를 직접 실행하지 않고, **"어떤 도구를 어떤 인자로 쓸지" 요청서(tool_calls)만 작성**

> LLM은 함수 실행 능력이 없음 — 호출 의도만 만들고, 실제 실행은 코드(Tool Node)가 담당

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """두 수를 더한다."""
    return a + b

llm_with_tools = llm.bind_tools([add])     # LLM에 도구 목록 알려줌
ai_msg = llm_with_tools.invoke("3 더하기 5는?")
print(ai_msg.tool_calls)
# [{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': ...}]  → 요청서만 생성
```

---

## Tool Node

LLM이 만든 요청서(tool_calls)를 받아 **실제 함수를 실행**하는 기성품 노드

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([add])     # 도구 목록으로 노드 생성
# AIMessage의 tool_calls를 읽어 함수 실행 → 결과를 ToolMessage로 반환
```

| 단계 | 주체 |
|---|---|
| 도구 호출 결정 | LLM (tool_calls 작성) |
| 도구 실제 실행 | ToolNode |
| 결과 전달 | ToolMessage로 State에 추가 |

---

## Routing

조건부 엣지로 State를 보고 **여러 갈래 중 하나를 선택**

| 방식 | 설명 |
|---|---|
| 규칙 기반 | State 값을 if 문으로 판단 |
| LLM 기반 | LLM이 분류한 결과로 분기 |

```python
def route(state: MessagesState):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else "end"   # 도구 호출 여부로 분기

builder.add_conditional_edges("llm", route, {"tools": "tool_node", "end": END})
```

---

## Parallelization

서로 **독립적인 여러 노드를 동시에 실행**

> 한 노드에서 여러 노드로 엣지를 펼치면 병렬 실행. 같은 칸에 쓰면 쌓기 리듀서 필수

```python
builder.add_edge("start", "search_web")     # 동시 실행
builder.add_edge("start", "search_db")       # 동시 실행
builder.add_edge("search_web", "merge")
builder.add_edge("search_db", "merge")       # 둘 다 끝나면 merge 진행
```

---

## Send

런타임에 **개수를 모를 때 노드를 동적으로 복제**해 병렬 실행 (Map-Reduce)

```python
from langgraph.types import Send

def fan_out(state):
    # 항목 개수만큼 process_item 노드를 동적 생성
    return [Send("process_item", {"item": x}) for x in state["items"]]

builder.add_conditional_edges("split", fan_out, ["process_item"])
```

> Parallelization은 노드 수가 고정, Send는 **데이터에 따라 가변적**으로 복제 (예: 문서 N개 각각 요약)

---

## Command

노드가 **"State 갱신 + 다음 행선지"를 한 번에 반환** — 노드가 라우팅까지 직접 결정

```python
from langgraph.types import Command
from typing import Literal

def node(state) -> Command[Literal["next_a", "next_b"]]:
    return Command(
        update={"answer": "..."},          # State 갱신
        goto="next_a",                      # 다음 노드 지정
    )
```

| 구분 | 조건부 엣지 | Command |
|---|---|---|
| 분기 결정 위치 | 별도 라우터 함수 | 노드 내부 |
| State 갱신 | 노드, 라우팅 따로 | 한 번에 |

---

## Loop

조건부 엣지가 **이전(자기) 노드를 가리켜** 형성되는 반복 흐름

```python
# work → check → (조건) → work (반복) 또는 END (탈출)
builder.add_conditional_edges("check", router, {"retry": "work", "done": END})
```

---

## Loop Termination

루프 탈출 — **조건 충족 + 물리적 한도** 두 겹의 안전장치

| 장치 | 방법 | 역할 |
|---|---|---|
| 논리적 종료 | 라우터가 `END` 반환 | 작업 완료 시 정상 탈출 |
| 물리적 한도 | `recursion_limit` | 무한 루프 방지 (강제 중단) |

```python
graph.invoke(input, {"recursion_limit": 25})   # 최대 25스텝 후 예외 발생
```

> 조건만 믿으면 무한 루프 위험 → recursion_limit으로 반드시 상한 설정

---

## ReAct Pattern

**LLM 노드 ↔ Tool 노드**를 잇고, 도구 호출이 없을 때까지 반복하는 에이전트 기본형

> ReAct = Reasoning(추론) + Acting(행동) — "생각 → 도구 사용 → 관찰"을 반복하며 답에 도달

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools=[add, search])
agent.invoke({"messages": [("user", "서울 날씨 알려줘")]})
```

```
LLM(추론) → tool_calls 있음? → Tool 실행 → LLM(관찰·추론) → ... → 없음 → END
```

> `tools_condition`(도구 호출 여부 판단 기성품)으로 LLM↔Tool 루프를 직접 구성할 수도 있음

---

## Functional API

`@task`(노드) + `@entrypoint`(전체)로 흐름을 **평범한 파이썬 코드로 작성**하는 Graph API의 대안

```python
from langgraph.func import task, entrypoint

@task
def step_a(x: int) -> int:
    return x + 1

@entrypoint()
def workflow(x: int):
    result = step_a(x).result()    # 일반 함수처럼 호출
    return result
```

| 구분 | Graph API | Functional API |
|---|---|---|
| 흐름 표현 | 노드·엣지 명시적으로 그림 | 일반 파이썬 제어문(if/for) |
| 가독성 | 구조가 시각적으로 드러남 | 익숙한 코드 스타일 |
| 적합 | 복잡한 분기·시각화 | 절차적 흐름 |
