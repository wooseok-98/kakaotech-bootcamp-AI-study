# 4-4. 고급 — 사람 개입과 스트리밍 (Interrupt · Human-in-the-Loop · Streaming)

그래프 실행을 **사람의 확인 지점에서 멈추고**, 진행 상황을 **실시간으로 흘려보내는** 기능

> 민감한 작업(결제·삭제) 전에 사람의 승인을 받고, 긴 작업은 중간 결과를 스트리밍해 UX를 개선

---

## Interrupt

그래프 실행을 **특정 지점에서 일시 정지**시키는 기능 — 사람 입력을 기다림

```python
from langgraph.types import interrupt, Command

def approval_node(state):
    decision = interrupt({"질문": "이 작업을 승인할까요?"})   # 여기서 멈춤
    return {"approved": decision}

# 실행 → interrupt에서 정지
graph.invoke(input, config)
# 사람 입력 후 재개 (interrupt 지점에 값 주입)
graph.invoke(Command(resume="yes"), config)
```

| 방식 | 설명 |
|---|---|
| `interrupt(...)` | 노드 안에서 동적으로 멈추고 입력 대기 (권장) |
| `interrupt_before/after` | compile 시 특정 노드 전후로 정적 중단점 지정 |

> interrupt는 checkpointer가 있어야 동작 — 멈춘 상태를 저장해야 재개 가능

---

## Human-in-the-Loop

실행 중간에 **사람이 검토·수정·승인**하도록 개입시키는 패턴 (HITL)

| 활용 | 설명 |
|---|---|
| Approve / Reject | 도구 실행·결제 전 사람 승인 |
| Edit State | 사람이 State(LLM 출력 등)를 직접 수정 후 재개 |
| Review Tool Call | LLM이 만든 도구 호출 인자를 검토·교정 |

```
LLM 제안 → [Interrupt] 사람 검토 → 승인/수정 → 재개 → 실행
```

> Interrupt(메커니즘) + 사람 판단을 결합한 패턴 — 신뢰성이 중요한 자동화에 필수

---

## Streaming

그래프 실행의 **중간 결과를 실시간으로 흘려보내는** 기능

```python
for chunk in graph.stream(input, config, stream_mode="updates"):
    print(chunk)        # 노드가 끝날 때마다 갱신 내용 출력
```

| stream_mode | 출력 내용 |
|---|---|
| `"values"` | 매 스텝의 전체 State |
| `"updates"` | 각 노드가 갱신한 부분만 |
| `"messages"` | LLM 토큰 단위 스트리밍 (챗봇 타이핑 효과) |
| `"custom"` | 노드 안에서 직접 내보낸 사용자 정의 데이터 |

```python
# LLM 토큰을 실시간으로 출력
for msg, meta in graph.stream(input, config, stream_mode="messages"):
    print(msg.content, end="", flush=True)
```

> 여러 모드를 리스트로 동시에 지정 가능 — `stream_mode=["updates", "messages"]`
