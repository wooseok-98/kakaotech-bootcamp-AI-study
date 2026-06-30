# LangGraph 한줄 정리

## AI Agent 개념

| 개념 | 한줄 정리 |
|--------|--------|
| LangGraph | State·Node·Edge로 LLM 워크플로우를 그래프로 표현하는 라이브러리 (LangChain의 순환·상태 한계를 보완) |
| Workflow | 다음 행동을 개발자가 코드로 미리 고정해 둔 흐름 (예측 가능, 기차) |
| AI Agent | 다음 행동을 LLM이 런타임에 스스로 결정하는 흐름 (유연, 택시) |
| Graph API | StateGraph + add_node + add_edge로 노드·엣지를 직접 그리는 방식 |

## 기초 구조

| 개념 | 한줄 정리 |
|--------|--------|
| StateGraph | 노드·엣지를 등록해 그래프를 조립하는 메인 빌더 |
| Graph State Schema | 그래프가 공유할 State의 칸(필드) 목록을 정의한 틀 (TypedDict) |
| Reducer | 칸에 새 값이 올 때 합치는 규칙 — 덮어쓰기(기본) vs 쌓기(`Annotated[list, operator.add]`) |
| MessagesState | messages 칸 + 쌓기 리듀서가 미리 세팅된 대화용 기성품 State |
| Node | State를 읽고 → 작업하고 → 바뀐 칸만 dict로 반환하는 함수 = 실제 업무 처리를 하는 단계 |
| Edge | 노드→노드 연결. 일반 엣지(고정) / 조건부 엣지(State 보고 분기·루프) |
| START | 그래프의 진입점을 나타내는 특수 노드 |
| END | 그래프 종료 지점을 나타내는 특수 노드 (루프 탈출구) |
| Compile | 설계도(StateGraph)를 실행 가능한 그래프 객체로 굳히는 단계 |
| Graph Visualization | 그래프 구조를 다이어그램(mermaid/ascii)으로 출력해 흐름을 확인·디버깅 |

## 핵심

| 개념 | 한줄 정리 |
|--------|--------|
| Tool Calling | LLM이 도구를 직접 실행하지 않고 "어떤 도구를 어떤 인자로 쓸지" 요청서(tool_calls)만 작성 |
| Tool Node | LLM이 만든 요청서를 받아 실제 함수를 실행하는 기성품 노드 (`ToolNode`) |
| Routing | 조건부 엣지로 State를 보고 여러 갈래 중 하나를 선택 (규칙 기반 / LLM 기반) |
| Parallelization | 서로 독립적인 여러 노드를 동시에 실행 (같은 칸에 쓰면 쌓기 리듀서 필수) |
| Send | 런타임에 개수를 모를 때 노드를 동적으로 복제해 병렬 실행 (Map-Reduce) |
| Command | 노드가 "State 갱신 + 다음 행선지"를 한 번에 반환 (노드가 라우팅까지 직접 결정) |
| Loop | 조건부 엣지가 이전(자기) 노드를 가리켜 형성되는 반복 흐름 |
| Loop Termination | 루프 탈출 — 조건 충족(`return END`) + 물리적 한도(`recursion_limit`) 두 겹 안전장치 |
| ReAct Pattern | LLM노드↔Tool노드를 `tools_condition`으로 잇고 도구 호출이 없을 때까지 반복하는 에이전트 기본형 |
| Functional API | `@task`(노드) + `@entrypoint`(전체)로 흐름을 평범한 파이썬 코드로 쓰는 Graph API의 대안 |

## 고급

| 개념 | 한줄 정리 |
|--------|--------|
| Persistence | State를 외부 저장소에 보존해 실행이 끝나도 이어 쓰는 기능 (멀티턴·재개의 토대) |
| Checkpointer | State 스냅샷을 저장·복원하는 객체 (MemorySaver/SqliteSaver/PostgresSaver) |
| Thread | 대화 세션을 구분하는 식별자(`thread_id`) — 세션별 State 분리 |
| Checkpoint | 각 스텝마다 자동 저장되는 State 스냅샷 하나 (1 thread = 여러 checkpoint) |
| Time Travel | 과거 체크포인트를 골라 그 상태로 되돌리거나 분기하는 기능 (디버깅·what-if) |
| Durable Execution | 중단돼도 마지막 체크포인트부터 재개해 끝난 단계는 건너뛰는 실행 방식 |
| Determinism | 같은 입력에 항상 같은 흐름을 따르는 성질 — 재개 시 그래프 경로 일관성 보장 |
| Idempotency | 같은 작업을 여러 번 해도 결과가 한 번과 같은 성질 (멱등성) — 중복 결제 방지 |
| Side Effect 처리 | 외부를 바꾸는 작업을 멱등하게/분리해 노드 재실행에 안전하게 다루는 것 |
| Short-term Memory | 한 thread(세션) 안에서 유지되는 단기 기억 — Checkpointer가 관리하는 State |
| Long-term Memory | 세션·thread를 넘어 지속되는 장기 기억 — Store에 namespace 단위로 저장 |
| Interrupt | 그래프 실행을 특정 지점에서 멈추고 사람 입력을 기다리는 기능 |
| Human-in-the-Loop | 실행 중간에 사람이 검토·수정·승인하도록 개입시키는 패턴 (HITL) |
| Streaming | 실행 중간 결과(State/노드 갱신/LLM 토큰)를 실시간으로 흘려보내는 기능 |
| Plan-and-Execute Pattern | 전체 계획을 먼저 세우고 단계를 하나씩 실행하며 계획을 갱신하는 패턴 |
| Evaluator-Optimizer Pattern | 생성 LLM과 평가 LLM이 생성↔비평을 반복해 결과를 개선하는 패턴 |
| Subgraph | 그래프를 하나의 노드처럼 다른 그래프에 포함시키는 모듈화 방식 |
| Multi-Agent System | 여러 전문 에이전트가 역할을 나눠 협업하는 시스템 (Supervisor/Network/계층형) |

