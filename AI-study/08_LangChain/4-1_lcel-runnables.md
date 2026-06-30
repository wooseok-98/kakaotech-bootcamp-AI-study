# 4-1. LCEL과 실행 조합 (Runnable · Passthrough · Parallel · Branch)

LCEL 체인을 구성하는 모든 컴포넌트의 공통 단위인 **Runnable** 과 이를 조합하는 도구들

> 프롬프트, 모델, 파서, 함수 등 모든 LangChain 컴포넌트는 Runnable — 같은 방식으로 실행·연결됨

---

## Runnable Interface

LangChain 컴포넌트를 **동일한 방식으로 실행**할 수 있게 하는 공통 인터페이스

| 메서드 | 설명 |
|---|---|
| `invoke(input)` | 입력 하나 실행 |
| `batch([...])` | 여러 입력 병렬 실행 |
| `stream(input)` | 청크 단위 스트리밍 |
| `ainvoke / abatch / astream` | 비동기 버전 |

```python
# 일반 함수도 RunnableLambda로 감싸면 체인에 넣을 수 있음
from langchain_core.runnables import RunnableLambda

length_runnable = RunnableLambda(lambda x: len(x))
chain = length_runnable | RunnableLambda(lambda n: n * 2)
print(chain.invoke("hello"))   # 10
```

> `|` 로 연결하면 자동으로 `RunnableSequence`가 됨

---

## RunnablePassthrough

입력을 **변경 없이 그대로 다음 단계로 전달**하는 Runnable

> 입력값을 보존하면서 동시에 다른 키를 추가할 때 주로 사용 (특히 RAG에서 question 전달용)

```python
from langchain_core.runnables import RunnablePassthrough

# assign: 기존 입력은 유지하고 새 키 추가
chain = RunnablePassthrough.assign(
    upper=lambda x: x["text"].upper()
)
print(chain.invoke({"text": "hello"}))
# {'text': 'hello', 'upper': 'HELLO'}
```

```python
# RAG 패턴: context는 검색, question은 그대로 통과
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
rag_chain.invoke("LangChain이 뭐야?")   # 입력이 question에 그대로 전달
```

---

## RunnableParallel

여러 Runnable을 **동시에 실행**하고 결과를 dict로 모아 반환

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    summary=prompt_summary | llm | StrOutputParser(),
    keywords=prompt_keywords | llm | StrOutputParser(),
)
result = parallel.invoke({"text": "긴 문서..."})
print(result["summary"], result["keywords"])
```

> 체인에서 `{"a": chain1, "b": chain2}` 형태의 dict는 자동으로 RunnableParallel로 변환됨

| 구분 | RunnableSequence (`|`) | RunnableParallel (dict) |
|---|---|---|
| 실행 방식 | 순차 (앞 출력 → 뒤 입력) | 병렬 (같은 입력을 동시에) |
| 결과 | 마지막 단계 출력 | 각 키별 결과 dict |

---

## RunnableBranch

조건에 따라 **서로 다른 Runnable을 선택 실행**하는 분기 처리 (if-elif-else)

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "코드" in x["topic"], code_chain),    # 조건1 → code_chain
    (lambda x: "번역" in x["topic"], translate_chain), # 조건2 → translate_chain
    default_chain,                                    # 그 외 → default
)
branch.invoke({"topic": "코드 리뷰 해줘"})
```

| 인자 | 의미 |
|---|---|
| `(조건함수, runnable)` | 조건이 True면 해당 runnable 실행 |
| 마지막 인자 | 어떤 조건도 안 맞을 때의 기본 runnable |

> 라우팅(routing): 질문 유형에 따라 다른 프롬프트·체인으로 보낼 때 활용
