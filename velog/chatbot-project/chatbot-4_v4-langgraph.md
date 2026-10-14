<!-- 📝 글 소개 (출간 시 입력): 포트폴리오용 AI 챗봇 프로젝트 4편 - LangGraph로 도구를 스스로 고르는 ReAct 에이전트를 만들며 워크플로우와 에이전트의 차이를 배우고, 도구 선택 튜닝을 삽질한 기록 -->

# 🎈 AI 학습 도우미 챗봇 만들기 (4) — 스스로 도구를 고르는 에이전트

[3편](여기에-3편-링크)까지는 질문이 오면 **무조건 노트와 위키를 둘 다 검색**했다. 흐름이 고정돼 있었다.

4편의 목표는 다르다. **LLM이 "어떤 도구를 쓸지 스스로 판단"하게 만드는 것.** 개념 질문이면 노트를, 최신 정보면 웹을 — 이걸 사람이 정하는 게 아니라 LLM이 결정한다. 이게 **에이전트**다.

___

## 📌 "이게 에이전트라고 할 수 있나?"

처음엔 v4를 "소스 라우터"로 계획했다. 질문을 보고 노트/위키 중 하나를 고르는 것. 그런데 만들기 전에 스스로 물었다. **"이걸 에이전트라고 부를 수 있나?"**

찾아보니 워크플로우와 에이전트는 다르다.

| | 워크플로우 | 에이전트 |
| --- | --- | --- |
| 흐름 | **개발자가 미리 정함** | **LLM이 런타임에 결정** |
| 판단 | 없거나 단발성 | 관찰→결정→행동을 **반복** |
| 도구 | 정해진 대로 | **LLM이 뭘 쓸지 고름** |

"소스 하나 고르기"는 **단발성 분기**라서 워크플로우에 가깝다. 진짜 에이전트가 되려면 LLM이 **도구를 스스로 고르고, 필요하면 반복하고, 언제 멈출지도 스스로** 정해야 한다.

그래서 방향을 바꿨다. **도구를 주고, LLM이 알아서 쓰는 ReAct 에이전트**로.

___

## 📌 도구 만들기 — `@tool`

에이전트에게 줄 도구 3개를 만들었다.

```python
@tool
def search_notes(query: str) -> str:
    """사용자가 정리한 학습 노트에서 검색한다. AI/ML 개념을 물을 때 사용."""
    return rag.search_notes(query)

@tool
def web_search(query: str) -> str:
    """웹을 검색한다. 노트·위키에 없는 최신·실시간 정보가 필요할 때 사용."""
    return _ddg.run(query)
```

`@tool` 데코레이터를 붙이면 함수가 **도구**가 된다. 여기서 중요한 건 **docstring**이다. 이 설명을 **LLM이 읽고** "이 질문엔 이 도구" 하고 판단한다. 즉 docstring이 곧 LLM의 도구 사용 설명서다.

그다음 LLM에 도구 목록을 쥐여준다.

```python
llm_with_tools = llm.bind_tools([search_notes, search_wiki, web_search])
```

이제 LLM은 답변 대신 **"search_notes('LSTM') 호출해줘"** 같은 지시를 만들 수 있다.

___

## 📌 ReAct 루프 — LangGraph

핵심은 **agent와 tools를 오가는 루프**다.

```python
builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)              # LLM
builder.add_node("tools", ToolNode(TOOLS))         # 도구 실행기
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # 도구? → tools / 답변? → END
builder.add_edge("tools", "agent")                 # 결과 갖고 다시 agent
```

```
START → agent(LLM) ──도구 원함?──▶ tools 실행 ──┐
             │  아니면(답변 준비)              │ (결과 갖고)
             ▼                                 │
            END  ◀──────────── agent 로 복귀 ──┘
```

- **agent**: LLM이 "도구를 부를지 / 최종 답변할지" 결정
- **tools_condition**: LLM 응답에 도구 호출이 있으면 tools로, 없으면 END로
- 이 루프를 **LLM이 스스로 돌린다.** 몇 번 검색할지, 언제 멈출지 LLM이 정한다

이게 v3와의 결정적 차이다. v3는 내가 짠 파이프대로 흘렀지만, v4는 **LLM이 운전대를 잡는다.**

___

## 📌 진짜 에이전트인지 — 궤적을 봐야 안다

에이전트를 만들고 나면 "정말 스스로 도구를 고르나?"를 확인해야 한다. 최종 답변만 봐선 모른다. 그래서 **LLM이 어떤 도구를 호출했는지 궤적을 출력**했다.

```
===== RAG가 뭐야? =====
🔧 도구 호출: ['search_notes', 'search_wiki']   ← 개념 → 노트+위키
💬 최종 답변: RAG는...

===== 오늘 서울 날씨 어때? =====
🔧 도구 호출: ['web_search']                    ← 실시간 → 웹
💬 최종 답변: 오늘 서울은...
```

질문에 따라 **다른 도구를 스스로 골랐다.** 개념은 노트·위키, 날씨는 웹으로. 이 궤적을 보는 순간 "아, 진짜 에이전트구나" 싶었다.

![에이전트 도구 선택 궤적](여기에-이미지-드래그)

___

## 🎈 시행 착오

### 📌 도구 선택이 처음엔 엉망이었다

첫 테스트에서 에이전트가 이상하게 굴었다.

- "RAG가 뭐야?" → 노트에 있는데 **웹 검색**을 했다
- "오늘 날씨?" → 웹 도구가 있는데 **안 쓰고 거절**했다

도구는 다 있는데 **선택을 못 했다.** 원인은 시스템 프롬프트가 도구 사용 기준을 충분히 안 알려준 것이었다. 프롬프트에 우선순위를 명시했다.

```
- AI/ML 개념 질문: 먼저 search_notes, 부족하면 search_wiki
- 최신·실시간 정보(날씨, 뉴스): 반드시 web_search
```

이 한 문단을 넣으니 도구 선택이 정상화됐다. **에이전트는 프롬프트로 "판단 기준"을 잡아줘야 한다**는 걸 배웠다.

### 📌 패키지 이름이 바뀌어 있었다

웹 검색으로 DuckDuckGo를 쓰려는데 `ModuleNotFoundError: No module named 'ddgs'`가 났다. `duckduckgo-search`를 깔았는데 LangChain은 `ddgs`를 찾았다. 알고 보니 **패키지가 `duckduckgo-search` → `ddgs`로 이름이 바뀌어** 있었다. `pip install ddgs`로 해결.

> 라이브러리 생태계는 계속 변한다. 에러 메시지를 그대로 읽으면(`install ddgs`) 대부분 답이 나와 있다.

___

## 📌 배운 점 + 마무리

- **워크플로우 vs 에이전트**: 흐름을 개발자가 정하면 워크플로우, **LLM이 정하면 에이전트**
- **도구 = 함수 + docstring.** docstring이 LLM의 판단 근거다
- **ReAct 루프**: agent ↔ tools 를 LLM이 스스로 반복·종료
- **궤적을 봐야 튜닝한다.** 도구 선택은 프롬프트로 잡아준다

v1(대화) → v2(RAG 직접) → v3(LangChain) → v4(에이전트)까지, 챗봇을 한 겹씩 쌓아 올렸다. 매 단계 "왜 이게 필요한가"를 손으로 확인한 게 가장 큰 수확이었다.

다음은 이 RAG가 **얼마나 잘하는지 평가(RAGAS)**하거나, **직접 파인튜닝한 모델**을 붙여보는 확장이다.

> 코드 전체: [github.com/wooseok-98/AI-study-chatbot](https://github.com/wooseok-98/AI-study-chatbot)
