# 1-1. AI Agent 개념 (LangGraph · Workflow · AI Agent · Graph API)

LLM의 작업 흐름을 **State·Node·Edge 그래프**로 표현해, 순환·분기·상태 관리를 가능하게 하는 라이브러리

> LCEL 체인은 한 방향(DAG)으로만 흐름 — 반복·조건 분기·상태 유지가 필요한 에이전트엔 부족해 LangGraph가 등장

---

## LangGraph

LangChain 위에서 동작하며, 작업 흐름을 **상태를 공유하는 노드들의 그래프**로 모델링하는 프레임워크

| LCEL 체인 | LangGraph |
|---|---|
| 단방향(DAG) 흐름 | 순환(loop)·분기 가능 |
| 상태 개념 약함 | State를 모든 노드가 공유 |
| 단순 파이프라인 | 에이전트·복잡한 제어 흐름 |

```bash
pip install langgraph
```

---

## Workflow

다음 행동을 **개발자가 코드로 미리 고정**해 둔 흐름

> 비유: **기차** — 정해진 선로(코드)대로만 움직임. 예측 가능하지만 경로 변경 불가

```
입력 → 검색 → 요약 → 출력   (순서가 코드에 고정됨)
```

---

## AI Agent

다음 행동을 **LLM이 런타임에 스스로 결정**하는 흐름

> 비유: **택시** — 목적지만 주면 LLM이 경로(도구 사용, 반복 여부)를 직접 판단. 유연하지만 예측이 어려움

| 구분 | Workflow | AI Agent |
|---|---|---|
| 행동 결정 주체 | 개발자(코드) | LLM(런타임) |
| 흐름 | 고정 | 동적 |
| 예측 가능성 | 높음 | 낮음 |
| 비유 | 기차 | 택시 |
| 적합 상황 | 절차가 명확한 작업 | 상황에 따라 판단이 필요한 작업 |

> 실제 시스템은 둘을 혼합 — 큰 틀은 Workflow로 고정하고 일부 단계만 Agent에게 위임하는 경우가 많음

---

## Graph API

`StateGraph`에 **노드와 엣지를 직접 등록**해 그래프를 그리는 LangGraph의 기본 방식

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    input: str
    output: str

def process(state: State):
    return {"output": state["input"].upper()}

builder = StateGraph(State)
builder.add_node("process", process)   # 노드 등록
builder.add_edge(START, "process")     # 진입 → process
builder.add_edge("process", END)       # process → 종료
graph = builder.compile()              # 실행 가능 그래프로 컴파일

print(graph.invoke({"input": "hello"}))   # {'input': 'hello', 'output': 'HELLO'}
```

| 구성 | 설명 |
|---|---|
| `StateGraph` | 노드·엣지를 등록하는 그래프 빌더 |
| `add_node` | 작업 단위(함수)를 노드로 등록 |
| `add_edge` | 노드 간 연결 정의 |
| `compile` | 설계도를 실행 가능한 객체로 변환 |

> 코드로 노드·엣지를 일일이 그리는 방식 — 흐름이 명시적으로 드러남. 대안으로 `Functional API`가 있음 ([[3-1_langgraph-core]] 참고)
