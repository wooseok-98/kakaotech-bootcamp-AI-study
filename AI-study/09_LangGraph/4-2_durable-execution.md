# 4-2. 고급 — 내구성 실행 (Durable Execution · Determinism · Idempotency · Side Effect)

실행이 **중단됐다 재개돼도 안전하게** 이어지도록 보장하는 실행 모델과 그 전제 조건들

> 체크포인트에서 재개할 때, 이미 끝난 작업을 다시 하거나(중복 결제) 결과가 달라지면 안 됨 → 이를 막는 규칙

---

## Durable Execution

실패·중단 후 **마지막 체크포인트부터 재개**해, 처음부터 다시 하지 않고 작업을 완수하는 실행 방식

> 긴 작업(여러 도구 호출, 사람 승인 대기) 중간에 끊겨도, 완료된 단계는 건너뛰고 이어서 진행

```python
# checkpointer가 있으면 중단 지점부터 재개
graph = builder.compile(checkpointer=checkpointer)
# 1차 실행 중 중단 → 같은 thread_id로 재호출하면 끝난 노드는 재실행 안 함
graph.invoke(None, config)
```

| durability 모드 | 설명 |
|---|---|
| `"exit"` | 종료 시에만 저장 (가장 빠름, 내구성 낮음) |
| `"async"` | 비동기 저장 (기본, 균형) |
| `"sync"` | 매 스텝 동기 저장 (가장 안전, 느림) |

---

## Determinism

같은 입력에 **항상 같은 흐름**을 따르도록 하는 성질 (결정성)

> 재개 시 그래프는 노드를 다시 거치며 흐름을 재구성 — 흐름이 매번 달라지면 재개가 깨짐

- 노드 함수 안에서 `random`, `time.now()` 같은 비결정적 값을 직접 쓰면 위험
- 비결정적 값이 필요하면 State에 저장해두고 재사용

---

## Idempotency

같은 작업을 **여러 번 해도 결과가 한 번 한 것과 같은** 성질 (멱등성)

> 재개로 노드가 재실행될 수 있으므로, 외부 작업(결제·메일)은 멱등하게 설계해야 중복 부작용 방지

```python
# 멱등성 키로 중복 실행 방지
def charge_node(state):
    pay_api.charge(amount=1000, idempotency_key=state["order_id"])
    # 같은 order_id로 두 번 호출돼도 한 번만 결제됨
    return {"paid": True}
```

| 성질 | 의미 | 깨지면 |
|---|---|---|
| Determinism | 흐름이 매번 동일 | 재개 시 그래프 경로 불일치 |
| Idempotency | 중복 실행해도 결과 동일 | 중복 결제·중복 전송 |

---

## Side Effect 처리

외부 세계를 바꾸는 작업(DB 쓰기, API 호출, 파일 저장)을 **재실행에 안전하게** 다루는 것

> 노드가 재실행될 수 있다는 전제 하에, 부작용을 분리하거나 멱등하게 만들어야 함

| 권장 사항 | 설명 |
|---|---|
| 부작용을 별도 노드로 분리 | 재실행 범위를 최소화 |
| 멱등성 키 사용 | 중복 호출 방어 |
| 결과를 State에 기록 | 재실행 시 "이미 했음" 판단 가능 |
| Functional API의 `@task` 활용 | task 결과가 체크포인트에 저장돼 재실행 시 캐시됨 |

```python
from langgraph.func import task

@task
def send_email(to: str):       # 한 번 성공하면 결과가 저장됨
    email_api.send(to)         # 재개 시 재호출되지 않고 캐시된 결과 반환
    return "sent"
```
