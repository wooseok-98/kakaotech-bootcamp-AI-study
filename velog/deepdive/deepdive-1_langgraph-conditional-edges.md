<!-- 📝 글 소개 (출간 시 입력): LangGraph의 Conditional Edges가 실행 흐름을 동적으로 제어하는 원리와, 조건을 잘못 설계했을 때 터지는 문제들 -->

# LangGraph Conditional Edges — 동적 흐름 제어와 오설계 시 실패 분석

> **딥다이브 주제:** LangGraph의 Conditional Edges가 그래프 실행 흐름을 동적으로 제어하는 방식을 설명하고, 조건 판단 기준이 모호하거나 잘못 설계되었을 때 실행 흐름에 어떤 문제가 발생할 수 있는지 서술하시오

---

## 들어가며 — 왜 "조건부" 엣지가 필요한가

LangGraph에서 노드를 잇는 가장 기본적인 방법은 고정 엣지(`add_edge`)다.

```python
builder.add_edge("retrieve", "generate")   # retrieve 다음은 언제나 generate
```

이 연결은 **그래프를 컴파일하는 시점에 행선지가 이미 정해진다.** 즉 실행 중에 무슨 일이 일어나든 `retrieve` 다음은 무조건 `generate`다.

하지만 실제 에이전트/RAG 파이프라인은 이런 판단을 끊임없이 해야 한다.

- 검색 결과가 부실하면 → **재검색**, 충분하면 → **답변 생성**
- LLM이 도구를 호출했으면 → **도구 실행**, 아니면 → **종료**
- 사용자 의도가 "환불"이면 → **환불 노드**, "배송"이면 → **배송 노드**

이 "그때그때 상황을 보고 갈 길을 정하는" 판단은 고정 엣지로는 표현할 수 없다. **실행 중(runtime)의 State를 읽고 다음 노드를 동적으로 결정하는 스위치**가 필요하고, 그것이 **Conditional Edge**다.

![](https://velog.velcdn.com/images/dan9872/post/e9ef84c0-0658-4e96-947f-dfac377268c8/image.png)

---


## 1. Conditional Edge가 흐름을 제어하는 원리

### 1-1. 세 가지 구성 요소

조건부 엣지는 `add_conditional_edges`로 등록하며, 세 조각으로 이뤄진다.

```python
builder.add_conditional_edges(
    "check",        # ① source: 분기가 일어나는 출발 노드
    router,         # ② path: State를 보고 다음 행선지를 반환하는 판단 함수
    {               # ③ path_map: 반환값 → 실제 노드 이름 매핑
        "retry": "work",
        "done": END,
    },
)
```

| 구성 요소 | 역할 |
|---|---|
| **source** | 이 노드 실행이 끝난 직후 분기 판단이 일어남 |
| **path (router)** | 현재 State를 입력받아 **다음 노드를 가리키는 키(문자열)를 반환** |
| **path_map** | 라우터가 반환한 키를 실제 노드 이름으로 변환하는 사전 |

> `path_map`을 생략하면 라우터가 반환한 문자열이 **곧바로 노드 이름**으로 쓰인다. 매핑을 두면 "판단 로직(키)"과 "그래프 배선(노드 이름)"을 분리할 수 있어 유지보수에 유리하다.

### 1-2. 라우터 함수 — 판단의 핵심

라우터는 **State를 받아 다음 노드 키를 반환하는 순수 함수**다.

```python
def router(state: State) -> str:
    if state["count"] < 3:		# 분기 조건
        return "retry"     		# → path_map을 거쳐 "work" 노드로
    return "done"          		# → END로
```

- 입력: 현재까지 누적된 **State**
- 출력: `path_map`의 **키 문자열** (또는 매핑이 없으면 노드 이름 자체)
- 이 함수는 **노드가 아니다.** State를 바꾸지 않고, 오직 "어디로 갈지"만 결정한다


### 1-3. "동적"의 의미 — 매 실행마다 새로 판단

핵심은 **판단 시점**이다. 그래프를 한 번 `invoke`할 때마다, source 노드가 끝나는 순간 라우터가 **그 시점의 State를 평가**해 분기를 정한다.

같은 그래프라도 입력이 달라지면 지나가는 경로가 달라진다. 이것이 "정적으로 그려진 그래프 위에서 동적으로 경로가 결정된다"는 뜻이다.

```
같은 그래프, 다른 실행 경로

입력 A (count=5)   START → check ──done──→ END
입력 B (count=1)   START → check ──retry─→ work → check ──retry─→ work → check ──done──→ END
```

### 1-4. 라우팅의 세 가지 유형

라우터가 무엇을 근거로 판단하느냐에 따라 크게 세 가지로 나뉜다.

| 유형 | 판단 근거 | 대표 예시 |
|---|---|---|
| **규칙 기반** | State 값을 조건문으로 비교 | `count < 3`, `len(docs) == 0` |
| **tool_calls 기반** | LLM 응답에 도구 호출이 있는지 | ReAct 에이전트 루프 |
| **LLM 분류 기반** | LLM이 생성한 라벨로 분기 | 사용자 의도 분류 라우팅 |

**tool_calls 기반 — ReAct의 심장**

```python
def route(state: MessagesState) -> str:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else "end"   # 도구 호출 여부로 분기

builder.add_conditional_edges("llm", route, {"tools": "tool_node", "end": END})
```

이 한 줄의 조건부 엣지가 `llm → tools → llm → tools → ... → end`라는 **자율 에이전트의 반복 루프**를 만든다. LangGraph의 prebuilt `create_react_agent`도 내부적으로 이 구조다.

### 1-5. 루프와 병렬 — 조건부 엣지의 확장

라우터가 반환하는 값의 **형태**를 바꾸면, 같은 조건부 엣지로 두 가지 확장 구조를 만들 수 있다.

**(1) 루프 — 자기 자신을 다시 가리키기**

조건부 엣지가 **이전(자기) 노드를 다시 가리키면** 반복 흐름이 된다. `END`를 반환하는 순간이 탈출구다. 앞서 본 재시도 그래프(`work → check → work → ...`)가 바로 이 형태다.

**(2) 병렬 fan-out — 여러 노드로 동시에 갈라지기**

**fan-out(팬아웃)** 은 하나의 노드에서 여러 갈래로 **동시에 퍼져나가 병렬 실행**하는 것을 말한다.

지금까지의 라우터는 키 하나(문자열)를 반환해 **다음 노드 하나**로 갔다. 그런데 라우터가 **리스트를 반환하면**, LangGraph는 리스트에 담긴 노드들을 **전부 동시에 실행**한다.

```
일반 분기 (하나 선택)          fan-out (여러 개 동시)

   check                          split
     │                          ┌───┼───┐
  "retry" 하나                node  node  node   ← 셋 다 병렬 실행
     ▼                          └───┼───┘
   work                          merge (다 끝나면 합류)
```

병렬로 갈라지는 방법은 두 가지이고, 결정적 차이는 **"몇 갈래로 갈라질지를 언제 아느냐"** 다.

| 방식 | 갈래 개수 | 예시 |
|---|---|---|
| 고정 엣지 여러 개 | **컴파일 때 이미 고정** | `search_web` + `search_db` 항상 2개 |
| `Send` 리스트 반환 | **런타임에 State 보고 결정** | 아이템이 3개면 3개, 100개면 100개 |

실행 대상을 런타임에 정하려면 **`Send` 객체 리스트**를 반환한다. `Send(노드이름, 그 노드에 넘길 State)` 형태로, "이 노드를 이 데이터로 실행해줘"를 아이템 개수만큼 만들어 던지는 것이다.

```python
# 동적 병렬: items 개수만큼 process_item을 동시 실행
def fan_out(state):
    # items = ["A", "B", "C"] 라면 → process_item이 A용·B용·C용 3개로 복제되어 병렬 실행
    return [Send("process_item", {"item": x}) for x in state["items"]]
```

![](https://velog.velcdn.com/images/dan9872/post/ef36c9dc-d333-4310-bbf3-5c3e5daba5ec/image.png)


---

## 2. 조건이 모호·오설계일 때의 문제

동적 제어의 힘은 곧 **판단 로직이 그래프의 버그 표면(bug surface)이 된다**는 뜻이다. 라우터가 잘못 판단하면 그래프 전체 흐름이 어긋난다. 판단 실패는 **증상 → 원인 → 처방** 관점으로 정리할 수 있다.

### 2-1. 실패 유형 진단 매트릭스

| 실패 양상 | 원인 | 실행 흐름에 미치는 결과 |
|---|---|---|
| **매핑 누락 (KeyError)** | 라우터가 `path_map`에 없는 키를 반환 | 실행 중단·에러 / 어디로도 못 감 |
| **무한 루프** | 종료 조건이 영영 참이 되지 않음 (State 미갱신 등) | `recursion_limit` 초과로 강제 중단 |
| **비결정 라우팅** | LLM 라우터가 모호·자유서술 라벨 생성 | 같은 입력에도 매번 다른 분기 → 재현 불가 |
| **조건 겹침** | 여러 조건이 동시에 참 / 우선순위 불명 | 의도치 않은 가지 선택, 앞선 `if`에 가려진 분기 |
| **도달 불가 노드** | 어떤 반환값도 특정 노드를 가리키지 않음 | 죽은 코드 — 그 기능이 영영 실행 안 됨 |
| **침묵 오라우팅** ⭐ | 에러 없이 **틀린 경로로 정상 진행** | 가장 잡기 어려움 — 결과만 이상하고 로그는 정상 |

### 2-2. 유형별 상세

**① 매핑 누락 — 판단은 됐는데 갈 곳이 없다**

라우터가 `"retry"`, `"done"`만 반환하도록 설계했는데 예외 상황에서 `"error"`를 반환하면, `path_map`에 `"error"` 키가 없어 흐름이 끊긴다. **라우터의 반환 가능 값 집합과 `path_map`의 키 집합이 정확히 일치**해야 한다.

**② 무한 루프 — 탈출 조건이 영영 참이 안 됨**

가장 흔한 오설계다. 재시도 루프를 만들었는데 정작 `count`를 증가시키는 코드가 없으면 조건이 영원히 `retry`로 남는다.

```python
def work(state):
    result = do_something(state)
    return {"result": result}          # ← count += 1 을 빠뜨림!

def router(state):
    return "retry" if state["count"] < 3 else "done"   # count가 늘지 않아 영원히 retry
```

LangGraph는 이런 폭주를 막기 위해 **`recursion_limit`(기본값 25)** 로 한 실행에서 밟을 수 있는 스텝 수에 상한을 둔다. 초과하면 `GraphRecursionError`를 던진다.

```python
graph.invoke(input, config={"recursion_limit": 10})
```

즉 상한은 **최후의 안전장치일 뿐, 종료 조건 설계의 대체물이 아니다.** 상한에 걸려 죽는 것 자체가 라우터 설계 결함의 신호다.


**③ 비결정 라우팅 — LLM 답변 유동성의 근본 리스크**


코드 에이전트에서 사용자의 자연어 요청을 도구별 경로로 분류하는 LLM 라우터를 생각해보자. `path_map`은 `{"run_tests": ..., "fix_bug": ..., "write_code": ...}` 세 키로 설계했다.

```python
def llm_router(state) -> str:
    prompt = f"다음 요청을 run_tests / fix_bug / write_code 중 하나로 분류: {state['request']}"
    return llm.invoke(prompt).content   # ← LLM이 뱉는 문자열을 그대로 반환
```

문제는 LLM이 **정확히 그 키 문자열만 뱉는다는 보장이 없다**는 것이다. 같은 "테스트 좀 돌려줘" 요청에도 모델은 이렇게 답할 수 있다.

- `"run_tests"` ✅ — 운 좋게 일치
- `"Run Tests"` / `"run the tests"` — 대소문자·표현 불일치 → `path_map` 키와 안 맞음
- `"테스트 실행"` — 아예 다른 언어
- ` "run_tests"라고 판단됩니다.` — 설명이 붙어 문자열이 오염됨

키와 한 글자라도 어긋나면 곧바로 ①번 **매핑 누락(KeyError)** 이 되고, temperature가 0이 아니면 **같은 입력에도 분기가 흔들려 재현이 안 된다.**

→ 처방: 프롬프트로 "한 단어로만 답해"라고 부탁하는 건 불안정하다. **structured output(Enum/Literal)으로 출력을 스키마 레벨에서 강제**해 반환값을 세 키로 못박는다.

```python
from pydantic import BaseModel
from typing import Literal

class Route(BaseModel):
    next: Literal["run_tests", "fix_bug", "write_code"]   # 이 셋 외엔 생성 불가

route = llm.with_structured_output(Route).invoke(state["request"])
return route.next   # 항상 path_map 키와 일치
```
**④ 조건 겹침 — 배타적이지 않은 분기**

```python
def router(state):
    if state["score"] > 0.5: return "accept"
    if state["score"] > 0.8: return "priority"   # ← 영원히 도달 못 함 (0.8은 이미 0.5 조건에 먹힘)
```

조건들이 **상호배타(mutually exclusive)** 이고 **전 경우를 빠짐없이(exhaustive)** 덮지 않으면, 앞 조건에 가려지거나(도달 불가) 어떤 조건에도 안 걸려(반환값 없음) 흐름이 깨진다. 조건 설계는 **MECE**여야 한다.
> **MECE**: Mutually Exclusive, Collectively Exhaustive
= "겹치지 않게(배타), 빠짐없이(전수)"

**⑤ 도달 불가 노드 — 조용히 사라진 기능**

`path_map`에 노드를 등록했지만 라우터가 그 키를 반환하는 경로가 실제로 없으면, 그 노드는 그래프에 그려져 있어도 절대 실행되지 않는다. 컴파일 에러가 나지 않기 때문에 **시각화로 눈으로 검증**하지 않으면 놓친다.

**⑥ 침묵 오라우팅 — 가장 위험한 실패**

에러도 없고 루프도 안 도는데, 라우터가 **틀린 가지를 선택**하는 경우다. 예를 들어 의도 분류 라우터가 "환불" 질문을 "일반 문의"로 잘못 분류하면, 그래프는 아무 문제 없이 끝까지 실행되지만 **사용자는 엉뚱한 답을 받는다.** 로그상 정상이라 탐지가 가장 어렵고, 이 때문에 **분기 추적(observability)** 이 필수다.

### 2-3. 예방·진단 도구 정리

| 도구 | 역할 | 막아주는 실패 |
|---|---|---|
| **그래프 시각화** (`draw_mermaid_png` / `draw_ascii`) | 컴파일된 그래프를 그림으로 뽑아, 어떤 노드로 가는 엣지가 없는지·배선이 의도대로인지 **눈으로 검증** | 도달 불가 노드, 배선 오류 |
| **default / fallback 분기** 두기 | 예상 못 한 반환값도 흘러갈 기본 목적지를 미리 등록해, 흐름이 끊기지 않게 **안전망** 제공 | 매핑 누락, 예외 반환 |
| **structured output (Enum/Literal)** | LLM 라우터의 출력을 스키마로 제약해, 반환값을 **`path_map` 키 집합으로 못박음** | 비결정 라우팅, 라벨 불일치 |
| **`recursion_limit` 상한** | 한 실행에서 밟을 수 있는 스텝 수에 상한을 걸어, 루프가 폭주해도 **강제 종료**시키는 최후의 브레이크 | 무한 루프 폭주 |
| **LangSmith 추적** | 각 라우터가 **어떤 State를 받아 어떤 값을 반환해 어디로 갔는지** 실행 기록을 남겨, 잘못된 분기를 사후에 추적 | 침묵 오라우팅, 분기 결정 근거 확인 |

![](https://velog.velcdn.com/images/dan9872/post/56e01438-327e-4a44-a6d3-b5e29736bd53/image.png)

---

## 3. 인접 개념과의 연결

**`Command` — 조건부 엣지의 대안**

노드가 "State 갱신 + 다음 행선지"를 **한 번에 반환**하는 방식이다. 별도 라우터 함수 없이 노드 안에서 분기를 결정한다.

| 구분 | 조건부 엣지 | Command |
|---|---|---|
| 분기 결정 위치 | 별도 라우터 함수 | 노드 내부 |
| State 갱신 | 노드와 라우팅이 따로 | 한 번에 |
| 적합한 경우 | 복잡한 분기·시각화 중요 | 절차적 흐름·라우팅과 갱신이 붙어있을 때 |

> 조건부 엣지는 **분기 로직을 그래프 구조로 드러내고 시각화**하기 좋고, `Command`는 **로직과 라우팅이 밀접할 때** 간결하다. ([[3-1_langgraph-core]] 참고)

---

## 4. 결론

- Conditional Edge는 **런타임 State를 읽어 다음 노드를 동적으로 고르는 스위치**이며, 이 하나로 분기·루프·병렬·자율 에이전트가 모두 표현된다.
- 그 유연함의 대가로, **판단 로직(라우터)이 곧 그래프의 버그 표면**이 된다. 조건이 모호하거나 배타적·전수적이지 않으면 무한 루프, 매핑 누락, 침묵 오라우팅 같은 흐름 붕괴가 발생한다.
- 안전한 설계 원칙 세 가지:
  1. **조건은 MECE로** — 상호배타·전수적. 반환값 집합 = `path_map` 키 집합
  2. **LLM 판단엔 가드레일** — structured output으로 출력 고정 + 종료 조건 명시 + `recursion_limit`
  3. **흐름은 반드시 관측** — 시각화로 도달성 검증, LangSmith로 분기 추적

> **다음 단계:** 조건부 엣지가 만드는 루프가 실제 자율 에이전트(ReAct, plan-and-execute)로 어떻게 확장되는지, 그리고 `interrupt`(human-in-the-loop)로 사람이 분기 판단에 개입하는 방식으로 이어진다. ([[4-4_human-in-the-loop]])

---
