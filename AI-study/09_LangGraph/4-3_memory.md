# 4-3. 고급 — 메모리 (Short-term Memory · Long-term Memory)

에이전트가 정보를 기억하는 두 층위 — **대화 한 세션 내 단기 기억**과 **세션을 넘는 장기 기억**

> 사람의 작업 기억(단기) vs 지식·경험(장기)에 대응 — 둘을 조합해야 자연스러운 에이전트가 됨

---

## Short-term Memory

**한 대화 세션(thread) 안에서** 유지되는 기억 — 체크포인터가 관리하는 State가 곧 단기 기억

> thread_id 단위로 보존 — 같은 대화 안의 이전 메시지를 기억하지만, 세션이 바뀌면 사라짐

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "session-1"}}

graph.invoke({"messages": [("user", "내 취미는 등산")]}, config)
graph.invoke({"messages": [("user", "내 취미가 뭐였지?")]}, config)
# 같은 thread → "등산" 기억 (단기 기억)
```

> 대화가 길어지면 토큰 한도 초과 → Message Trimming·요약으로 관리 ([[5-1_message-history]] 참고)

---

## Long-term Memory

**여러 세션·thread를 넘어** 지속되는 기억 — `Store`에 저장하고 필요할 때 검색

> 사용자 선호, 누적된 지식처럼 대화가 끝나도 남아야 하는 정보를 저장

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
graph = builder.compile(checkpointer=checkpointer, store=store)

# 노드 안에서 store 접근 (namespace로 사용자별 구분)
def node(state, *, store):
    ns = ("user-1", "memories")
    store.put(ns, "hobby", {"value": "등산"})     # 저장
    items = store.search(ns)                        # 검색
    return {...}
```

| 구분 | Short-term Memory | Long-term Memory |
|---|---|---|
| 범위 | 한 thread(세션) 내 | thread·세션을 초월 |
| 저장소 | Checkpointer (State) | Store |
| 단위 | thread_id | namespace (예: 사용자 id) |
| 예시 | 현재 대화 맥락 | 사용자 프로필·선호·누적 지식 |

> Store는 의미 검색(임베딩)도 지원 — 관련 기억만 골라 프롬프트에 주입하는 식으로 활용
