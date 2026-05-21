# Query String vs Path Variable

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