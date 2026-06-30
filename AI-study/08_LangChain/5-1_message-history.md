# 5-1. 대화 이력 처리 (RunnableWithMessageHistory · Message Trimming)

체인이 **이전 대화를 기억**하게 만들고, 길어진 기록을 **잘라내 토큰 한도를 관리**하는 도구

> LLM은 상태가 없어(stateless) 매 호출이 독립적 — 대화 맥락을 유지하려면 기록을 직접 주입해야 함

---

## RunnableWithMessageHistory

체인에 **대화 기록을 자동으로 주입·저장**해주는 래퍼 Runnable

> 세션 ID별로 기록을 분리 관리 — 사용자마다 독립된 대화 유지

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 친절한 비서야."),
    MessagesPlaceholder(variable_name="history"),   # 기록이 주입될 자리
    ("human", "{input}"),
])
chain = prompt | llm

store = {}   # 세션별 기록 저장소
def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chat = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",      # 사용자 입력 키
    history_messages_key="history",  # Placeholder 변수명과 일치
)

config = {"configurable": {"session_id": "user-1"}}
chat.invoke({"input": "내 이름은 우석이야"}, config=config)
print(chat.invoke({"input": "내 이름이 뭐야?"}, config=config).content)
# "우석님이에요" → 이전 대화 기억
```

| 인자 | 역할 |
|---|---|
| `get_session_history` | session_id로 기록 객체 반환하는 함수 |
| `input_messages_key` | 입력 dict에서 사용자 메시지 키 |
| `history_messages_key` | MessagesPlaceholder의 변수명 |
| `config`의 `session_id` | 대화 세션 구분자 |

---

## Message Trimming

토큰 한도를 넘지 않도록 **오래된 메시지를 잘라내는** 기능

> 대화가 길어지면 토큰 비용·한도 초과 발생 — 최근 N개 또는 N토큰만 유지

```python
from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",          # 최근 메시지부터 유지
    token_counter=llm,        # 토큰 수 계산 기준 모델
    include_system=True,      # SystemMessage는 항상 보존
    start_on="human",         # human 메시지부터 시작하도록 정렬
)

trimmed = trimmer.invoke(messages)   # 한도 내 메시지 목록 반환
```

| 매개변수 | 설명 |
|---|---|
| `max_tokens` | 유지할 최대 토큰 수 |
| `strategy` | `"last"`(최근 우선) / `"first"`(앞쪽 우선) |
| `include_system` | 시스템 메시지 보존 여부 |
| `token_counter` | 토큰 계산에 쓸 모델 또는 함수 |

```python
# 체인 앞단에 trimmer를 끼워 자동 적용
chain = (
    RunnablePassthrough.assign(history=lambda x: trimmer.invoke(x["history"]))
    | prompt | llm
)
```

| 전략 | 장점 | 단점 |
|---|---|---|
| Trimming (잘라내기) | 간단, 빠름 | 잘린 부분의 맥락 손실 |
| Summarization (요약) | 맥락 보존 | 추가 LLM 호출 비용 |
