# 2-2. 요청 데이터 (Query String, Path Variable, JSON)

---

## Query String vs Path Variable

URL로 데이터를 전달하는 두 가지 방식

| 구분 | 형태 | 주요 용도 |
| --- | --- | --- |
| Query String | `/posts?category=books&page=1` | 검색, 필터링, 정렬, 페이지네이션 |
| Path Variable | `/posts/123` | 특정 리소스 식별 |

---

## Query String

URL 끝에 `?`를 붙이고 `key=value` 형태로 전달, 여러 개는 `&`로 구분

```
GET /posts?category=books&search=coding
GET /posts?offset=0&limit=10
```

### FastAPI에서 사용

```python
@app.get("/posts")
def get_posts(category: str, page: int = 1):
    ...
```

파라미터에 기본값이 있으면 선택, 없으면 필수

**값 범위 검증이 필요한 경우**

```python
from fastapi import Query

@app.get("/posts")
def get_posts(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=50),
):
    ...
```

| 옵션 | 의미 |
| --- | --- |
| `ge` | 이상 (greater than or equal) |
| `le` | 이하 (less than or equal) |

조건 불만족 시 자동으로 `422` 반환

---

## Path Variable

URL 경로에 값을 직접 포함

```
GET /posts/100
DELETE /users/1
```

### FastAPI에서 사용

```python
@app.get("/posts/{post_id}")
def get_post(post_id: int):
    ...
```

타입 힌트만으로 자동 검증 — `/posts/abc` 요청 시 `422` 자동 반환

---

## JSON (JavaScript Object Notation)

클라이언트와 서버 간 데이터를 주고받는 표준 텍스트 포맷  
언어에 독립적으로 대부분의 프로그래밍 언어에서 지원

---

## 기본 문법

```json
{
    "name": "wayne",
    "age": 25,
    "hobbies": ["reading", "hiking"]
}
```

- 키는 반드시 `"문자열"` 형태
- 키-값은 `:` 로 구분, 각 쌍은 `,` 로 구분
- 객체는 `{}`, 배열은 `[]`

**배열 예시**

```json
[
    {"id": 1, "name": "apple"},
    {"id": 2, "name": "banana"}
]
```

---

## FastAPI에서 JSON 응답

### dict / list 반환 (기본)

```python
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "pong"}  # 자동으로 JSON 변환
```

### JSONResponse (상태 코드 직접 지정)

```python
from fastapi.responses import JSONResponse

@app.get("/items")
def get_items():
    data = [{"id": 1, "name": "apple"}]
    return JSONResponse(content=data, status_code=201)
```

### 커스텀 메시지 포함

```python
return JSONResponse(
    content={
        "message": "Created Successfully",
        "data": data,
    },
    status_code=201,
)
```

---

## 데이터 포맷 비교

| 포맷 | 특징 | 주요 사용처 |
| --- | --- | --- |
| JSON | 가볍고 읽기 쉬움, 대부분 언어 지원 | 웹 API, 설정 파일 |
| XML | 태그 기반, 복잡한 구조 표현 가능 | 레거시 시스템, 문서 교환 |
| CSV | 구조 단순, 대량 데이터에 효율적 | 엑셀, 데이터 분석 |
