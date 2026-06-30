# 4-1. 고급 — 영속성 (Persistence · Checkpointer · Thread · Checkpoint · Time Travel)

그래프 실행 상태를 **저장**해 대화를 이어가고, 과거 시점으로 되돌아갈 수 있게 하는 기능

> 기본 그래프는 실행이 끝나면 State가 사라짐 — 영속성을 붙여야 멀티턴 대화·중단 후 재개가 가능

---

## Persistence

그래프의 State를 **외부 저장소에 보존**해, 실행이 끝나도 다음 호출에서 이어 쓰는 기능

> 멀티턴 대화, Human-in-the-Loop, Time Travel 등 고급 기능의 토대

---

## Checkpointer

State 스냅샷을 **저장·복원하는 객체** — compile 시 주입

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()                 # 메모리 기반 (개발용)
graph = builder.compile(checkpointer=checkpointer)
```

| 종류 | 저장 위치 | 용도 |
|---|---|---|
| `MemorySaver` | 메모리 | 개발·테스트 |
| `SqliteSaver` | SQLite 파일 | 로컬 영구 저장 |
| `PostgresSaver` | PostgreSQL | 운영 환경 |

---

## Thread

**대화 세션을 구분하는 식별자** — `thread_id`로 사용자/세션별 State를 분리

```python
config = {"configurable": {"thread_id": "user-1"}}

graph.invoke({"messages": [("user", "내 이름은 우석")]}, config)
graph.invoke({"messages": [("user", "내 이름은?")]}, config)
# 같은 thread_id → 이전 State 이어받아 "우석" 기억
```

> thread_id가 다르면 완전히 독립된 대화 — 8장 `RunnableWithMessageHistory`의 session_id와 같은 역할

---

## Checkpoint

특정 시점의 **State 스냅샷 하나** — 각 스텝마다 자동 저장됨

```python
# 현재 State 조회
state = graph.get_state(config)
print(state.values)          # State 값
print(state.next)            # 다음 실행될 노드

# 전체 체크포인트 이력 조회
for ckpt in graph.get_state_history(config):
    print(ckpt.config, ckpt.values)
```

| 구분 | Thread | Checkpoint |
|---|---|---|
| 단위 | 대화 세션 전체 | 세션 내 한 스텝의 스냅샷 |
| 개수 | 1 세션 = 1 thread | 1 thread = 여러 checkpoint |

---

## Time Travel

저장된 체크포인트 중 **과거 시점을 선택해 그 상태로 되돌리거나 분기**하는 기능

```python
# 특정 체크포인트 id로 그 시점부터 재실행
past_config = {"configurable": {"thread_id": "user-1", "checkpoint_id": "..."}}
graph.invoke(None, past_config)        # 과거 상태에서 다시 진행

# State를 수정해 분기(다른 경로 탐색)
graph.update_state(past_config, {"answer": "수정된 값"})
```

| 활용 | 설명 |
|---|---|
| 디버깅 | 문제가 생긴 시점으로 돌아가 원인 분석 |
| 분기 탐색 | 과거 상태를 바꿔 다른 결과 시도 (what-if) |
| 복구 | 잘못된 스텝 이전으로 롤백 |
