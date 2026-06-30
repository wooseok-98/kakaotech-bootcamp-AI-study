# 4-5. 고급 — 에이전트 패턴 (Plan-and-Execute · Evaluator-Optimizer · Subgraph · Multi-Agent)

복잡한 작업을 **계획·검증·모듈화·역할 분담**으로 풀어내는 대표적인 에이전트 설계 패턴들

> 단일 ReAct 루프로 부족한 큰 작업을, 구조화된 워크플로우로 나눠 신뢰성과 확장성을 확보

---

## Plan-and-Execute Pattern

먼저 **전체 계획(단계 목록)을 세우고**, 단계를 하나씩 실행하며 필요 시 계획을 갱신

> ReAct는 매 스텝 즉흥적으로 판단 — Plan-and-Execute는 큰 그림을 먼저 그려 길 잃음을 방지

```
Planner(계획 수립) → Executor(단계 실행) → Replan(남은 계획 갱신) → ... → 완료
```

| 노드 | 역할 |
|---|---|
| Planner | 목표를 하위 단계 목록으로 분해 |
| Executor | 각 단계를 실행 (도구 호출 등) |
| Replanner | 결과를 보고 계획 수정·다음 단계 결정 |

> 장점: 긴 작업에서 일관성 유지, 토큰 절약 / 단점: 계획 단계의 오류가 전파될 수 있음

---

## Evaluator-Optimizer Pattern

한 LLM이 **생성**하고, 다른 LLM이 **평가·피드백**해 결과를 반복 개선 (생성↔비평 루프)

> 초안 → 비평 → 수정을 반복 — 번역·글쓰기·코드 등 품질을 점진적으로 끌어올리는 작업에 적합

```
Generator(생성) → Evaluator(평가) → 통과? → 종료
                       ↑__________ 미통과(피드백) ___|
```

| 노드 | 역할 |
|---|---|
| Generator(Optimizer) | 결과 생성·피드백 반영해 개선 |
| Evaluator | 기준 충족 여부 판정 + 피드백 제공 |

> Loop + 조건부 엣지로 구현 — Evaluator가 "통과"를 반환할 때까지 반복 (Loop Termination으로 상한 설정)

---

## Subgraph

**그래프를 하나의 노드처럼** 다른 그래프 안에 포함시키는 모듈화 방식

> 복잡한 흐름을 작은 그래프로 캡슐화해 재사용 — 큰 시스템을 부품 단위로 조립

```python
sub_builder = StateGraph(SubState)
# ... 서브그래프 구성 ...
subgraph = sub_builder.compile()

parent = StateGraph(State)
parent.add_node("sub", subgraph)     # 컴파일된 그래프를 노드로 추가
```

| 상태 공유 | 방식 |
|---|---|
| State 스키마 공유 | 부모·자식이 같은 칸 → 자동 연동 |
| 스키마 다름 | 변환 함수로 입출력 매핑 |

> 멀티 에이전트에서 각 에이전트를 subgraph로 구성하는 데 자주 쓰임

---

## Multi-Agent System

여러 **전문 에이전트가 역할을 나눠 협업**하는 시스템

> 하나의 거대 에이전트보다, 역할별로 분리된 작은 에이전트들이 관리·성능 면에서 유리

| 구조 | 설명 |
|---|---|
| Supervisor | 관리자 에이전트가 작업을 분배하고 결과를 취합 |
| Network | 에이전트들이 서로 직접 핸드오프(handoff)하며 협업 |
| Hierarchical | 관리자-팀 계층 구조로 확장 |

```
       ┌─ Supervisor ─┐         (작업 분배·취합)
       │      │       │
   Researcher  Coder  Writer    (전문 에이전트)
```

```python
# Command의 goto로 에이전트 간 제어 이양(handoff)
def supervisor(state) -> Command:
    return Command(goto="researcher", update={...})   # 다음 에이전트로 넘김
```

| 패턴 | 핵심 |
|---|---|
| Plan-and-Execute | 계획 후 단계 실행 |
| Evaluator-Optimizer | 생성↔비평 반복 개선 |
| Subgraph | 그래프 모듈화·재사용 |
| Multi-Agent | 역할 분담 협업 |
