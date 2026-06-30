# 3-1. 출력 구조화 (Structured Output · JsonOutputParser · PydanticOutputParser)

LLM의 자유 형식 텍스트 응답을 **JSON·객체 등 정해진 구조**로 받아내는 기능

> 응답을 코드에서 바로 사용하려면 일정한 형식이 필요 — 후속 로직(DB 저장, API 응답)과 연결하기 위함

---

## Structured Output

`with_structured_output()` 으로 모델이 **스키마에 맞는 객체를 직접 반환**하게 하는 기능

> 가장 권장되는 방법 — 모델이 스키마를 인지하고 파싱 실패 없이 구조화된 결과를 돌려줌

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class Movie(BaseModel):
    title: str = Field(description="영화 제목")
    year: int = Field(description="개봉 연도")
    genre: str = Field(description="장르")

llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(Movie)

result = structured_llm.invoke("영화 인셉션 정보 알려줘")
print(result.title, result.year)   # Movie 객체로 바로 반환
```

| 방식 | 특징 |
|---|---|
| `with_structured_output` | 모델 내장 기능 사용, 파서 불필요, 가장 안정적 |
| OutputParser | 모델 출력 텍스트를 사후 파싱, 프롬프트에 형식 지시 필요 |

---

## JsonOutputParser

LLM 응답을 **JSON(dict)** 형태로 파싱하는 출력 파서

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template(
    "다음 질문에 JSON으로 답해줘.\n{format_instructions}\n질문: {query}"
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
result = chain.invoke({"query": "한국의 수도와 인구는?"})
print(result)   # {'capital': '서울', 'population': ...}  → dict
```

> `get_format_instructions()`: 모델에게 "JSON으로 출력하라"는 지시문을 자동 생성해 프롬프트에 주입

---

## PydanticOutputParser

**Pydantic 모델**을 기준으로 응답을 검증하고 **객체로 변환**하는 출력 파서

> JsonOutputParser와 달리 타입 검증·필드 제약을 적용하고, dict가 아닌 객체를 반환

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="이름")
    age: int = Field(description="나이")

parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_template(
    "{format_instructions}\n{query}"
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
person = chain.invoke({"query": "홍길동은 30살이야"})
print(person.name, person.age)   # Person 객체 (타입 검증됨)
```

### 파서 비교

| 파서 | 반환 타입 | 검증 | 용도 |
|---|---|---|---|
| `StrOutputParser` | str | 없음 | 단순 텍스트 추출 |
| `JsonOutputParser` | dict | 형식만 | 가벼운 JSON 응답 |
| `PydanticOutputParser` | Pydantic 객체 | 타입·제약 | 엄격한 스키마 검증 |

> 최신 LangChain에서는 대부분 `with_structured_output`으로 대체 가능하나, 모델이 해당 기능을 지원하지 않을 때 파서 방식을 사용
