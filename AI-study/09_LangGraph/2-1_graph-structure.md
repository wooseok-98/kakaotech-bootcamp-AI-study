# 2-1. LangGraph 기초 구조 (State · Node · Edge · Compile)

그래프를 구성하는 **상태(State) · 노드(Node) · 엣지(Edge)** 와, 이를 조립·실행 가능한 형태로 만드는 요소들

> State = 노드들이 공유하는 데이터, Node = 작업 단위, Edge = 노드 간 연결. 이 셋으로 모든 흐름을 표현

---

## StateGraph

노드·엣지를 등록해 그래프를 **조립하는 메인 빌더**

```python
from langgraph.graph import StateGraph
builder = StateGraph(State)        # State 스키마를 기준으로 생성
```

| 메서드 | 역할 |
|---|---|
| `add_node(name, fn)` | 노드(작업 함수) 등록 |
| `add_edge(a, b)` | 고정 연결 |
| `add_conditional_edges(...)` | 조건부 분기 연결 |
| `compile()` | 실행 가능한 그래프 생성 |

---

## Graph State Schema

그래프가 공유할 **State의 칸(필드) 목록**을 정의한 틀 — 보통 `TypedDict`로 작성

> 모든 노드는 이 State를 읽고, 바뀐 칸만 dict로 반환 → State가 갱신됨

```python
from typing_extensions import TypedDict

class State(TypedDict):
    question: str      # 칸 1
    answer: str        # 칸 2
    count: int         # 칸 3
```

---

## Reducer

칸에 **새 값이 올 때 기존 값과 합치는 규칙**

| 방식 | 동작 | 지정법 |
|---|---|---|
| 덮어쓰기 (기본) | 새 값으로 교체 | 일반 타입 (`str`, `int`) |
| 쌓기 (누적) | 리스트에 추가 | `Annotated[list, operator.add]` |

```python
from typing import Annotated
import operator

class State(TypedDict):
    answer: str                                  # 덮어쓰기
    logs: Annotated[list, operator.add]          # 쌓기 (병렬 노드 결과 누적)
```

> 병렬 노드가 같은 칸에 동시에 쓰면 덮어쓰기는 충돌 → 쌓기 리듀서 필수 ([[3-1_langgraph-core]] Parallelization 참고)

---

## MessagesState

`messages` 칸 + 메시지 누적 리듀서(`add_messages`)가 **미리 세팅된 대화용 기성품 State**

```python
from langgraph.graph import MessagesState
# class MessagesState(TypedDict):
#     messages: Annotated[list, add_messages]   # 내부 정의

def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}   # 자동 누적
```

> `add_messages`는 단순 추가뿐 아니라 같은 id의 메시지는 갱신, 새 메시지는 append하는 똑똑한 리듀서

---

## Node

State를 **읽고 → 작업하고 → 바뀐 칸만 dict로 반환**하는 함수 (실제 업무 처리 단계)

```python
def my_node(state: State):
    new_answer = state["question"] + "에 대한 답"
    return {"answer": new_answer}   # 바뀐 칸만 반환 (나머지는 유지)
```

> 반환한 dict의 키만 리듀서를 거쳐 State에 반영됨

---

## Edge

노드와 노드를 **연결**하는 선

| 종류 | 설명 |
|---|---|
| 일반 엣지 | 고정 연결 — `add_edge("a", "b")` |
| 조건부 엣지 | State를 보고 분기·루프 — `add_conditional_edges("a", router_fn)` |

```python
def router(state: State):
    return "retry" if state["count"] < 3 else "end"   # 다음 노드 이름 반환

builder.add_conditional_edges("check", router, {"retry": "work", "end": END})
```

---

## START / END

그래프의 **진입점**과 **종료점**을 나타내는 특수 노드

| 노드 | 역할 |
|---|---|
| `START` | 그래프 실행이 시작되는 지점 |
| `END` | 그래프가 종료되는 지점 (루프 탈출구) |

```python
from langgraph.graph import START, END
builder.add_edge(START, "first_node")
builder.add_edge("last_node", END)
```

---

## Compile

설계도(StateGraph)를 **실행 가능한 그래프 객체로 굳히는** 단계

```python
graph = builder.compile()
graph.invoke({"question": "..."})    # invoke/stream/batch 사용 가능
```

> compile 시 checkpointer·interrupt 등 실행 옵션을 함께 지정 ([[4-1_persistence]] 참고)

---

## Graph Visualization

그래프 구조를 **다이어그램으로 출력**해 흐름을 확인·디버깅

```python
# Mermaid 다이어그램 (Jupyter)
from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())

# 텍스트(ASCII)
print(graph.get_graph().draw_ascii())
```

> 노드·엣지·분기가 의도대로 연결됐는지 눈으로 검증할 때 유용
