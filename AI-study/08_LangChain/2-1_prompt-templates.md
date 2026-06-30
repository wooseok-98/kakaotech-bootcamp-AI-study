# 2-1. 프롬프트 구성 (PromptTemplate · ChatPromptTemplate · FewShot · MessagesPlaceholder)

변수 값을 채워 넣어 **동적으로 프롬프트를 생성**하는 템플릿 도구들

> 프롬프트를 코드에 하드코딩하지 않고, 재사용 가능한 틀로 분리

---

## PromptTemplate

`{변수}` 자리표시자를 가진 **단일 문자열 프롬프트** 템플릿

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("{product}의 장점 3가지를 알려줘")
print(prompt.format(product="전기차"))
# "전기차의 장점 3가지를 알려줘"
```

| 메서드 | 설명 |
|---|---|
| `from_template(str)` | 문자열에서 변수 자동 추출 |
| `format(**kwargs)` | 변수를 채워 완성된 문자열 반환 |
| `invoke(dict)` | LCEL 체인에서 사용하는 실행 방식 |

> 주로 일반 LLM(비대화형)이나 단순 텍스트 생성에 사용

---

## ChatPromptTemplate

여러 **역할별 메시지(System/Human/AI)** 구조를 가진 프롬프트 템플릿 — 대화형 모델에 사용

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 {language} 전문 번역가야."),
    ("human", "다음 문장을 번역해줘: {text}"),
])

messages = prompt.invoke({"language": "영어", "text": "안녕하세요"})
print(messages.to_messages())   # [SystemMessage, HumanMessage]
```

| 튜플 형식 | 대응 메시지 |
|---|---|
| `("system", "...")` | SystemMessage |
| `("human", "...")` | HumanMessage |
| `("ai", "...")` | AIMessage |

```python
# LCEL 체인에 바로 연결
chain = prompt | llm | StrOutputParser()
print(chain.invoke({"language": "영어", "text": "안녕하세요"}))
```

---

## FewShotPromptTemplate

**예시(Example)** 몇 개를 프롬프트에 넣어 모델이 원하는 출력 형식을 따라하게 만드는 템플릿

> Few-shot Learning: 예시를 통해 모델이 패턴을 모방하도록 유도

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

examples = [
    {"input": "행복", "output": "happy"},
    {"input": "슬픔", "output": "sad"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "한국어 단어를 영어로 번역해."),
    few_shot,                       # 예시들이 여기 삽입됨
    ("human", "{input}"),
])
print(final_prompt.invoke({"input": "기쁨"}).to_messages())
```

| 구분 | Zero-shot | Few-shot |
|---|---|---|
| 예시 | 없음 | 몇 개 제공 |
| 형식 일관성 | 낮음 | 높음 (예시 형식 모방) |

---

## MessagesPlaceholder

이전 대화 기록 같은 **메시지 목록을 통째로 끼워 넣는** 자리표시자

> 길이가 가변적인 대화 히스토리를 프롬프트의 특정 위치에 동적으로 삽입할 때 사용

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 친절한 비서야."),
    MessagesPlaceholder(variable_name="history"),   # 대화 기록 자리
    ("human", "{input}"),
])

messages = prompt.invoke({
    "history": [
        HumanMessage(content="내 이름은 우석이야"),
        AIMessage(content="반가워요 우석님!"),
    ],
    "input": "내 이름이 뭐였지?",
})
```

> `RunnableWithMessageHistory`와 결합하면 대화 기록을 자동으로 이 자리에 주입 ([[5-1_message-history]] 참고)
