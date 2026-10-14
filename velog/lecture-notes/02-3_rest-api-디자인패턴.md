<!-- 📝 글 소개 (출간 시 입력): REST API 설계 원칙과 Route-Controller-Model 디자인 패턴 정리 -->

# 🎈 REST API와 디자인 패턴

**REST API**는 HTTP 메서드와 명사형 URL로 자원을 표현하고 조작하는 설계 방식이다.

REST의 핵심 제약 중 하나는 **무상태(Stateless)**다. 서버는 클라이언트의 이전 요청 상태를 저장하지 않으므로, 각 요청은 인증 토큰 같은 **필요한 정보를 매번 포함**해야 한다. 이 덕분에 어느 서버가 요청을 받아도 동일하게 처리되어, 서버를 여러 대로 늘리는 **스케일 아웃(scale-out)**이 쉬워진다.

___

## 📌 REST 원칙

| 원칙 | 설명 |
| --- | --- |
| **Uniform Interface** | 일관된 URL 구조와 HTTP 메서드 사용 |
| **Client-Server** | 클라이언트와 서버를 독립적으로 분리 |
| **Stateless** | 서버는 클라이언트 상태를 저장하지 않고, 요청마다 필요한 정보를 모두 포함 |
| **Cacheable** | 응답은 캐싱 가능 여부를 명시 |

___

## 📌 URL 설계 규칙

URL은 **행위가 아닌 자원(명사)**으로 표현하고, 행위는 HTTP 메서드로 나타낸다.

| 규칙 | 좋은 예 | 나쁜 예 |
| --- | --- | --- |
| URI는 명사로 표현 | `/users` | `/getUsers` |
| 행위는 메서드로 표현 | `POST /users` | `GET /createUser` |
| 계층 관계는 `/`로 표현 | `/users/1/posts` | `/users-posts-1` |
| 소문자 사용 | `/users` | `/Users` |
| 하이픈 사용, 밑줄 금지 | `/user-posts` | `/user_posts` |
| URI 끝 슬래시 금지 | `/users` | `/users/` |

| 기능 | 올바른 설계 |
| --- | --- |
| 목록 조회 | `GET /users` |
| 단건 조회 | `GET /users/1` |
| 생성 | `POST /users` |
| 전체 수정 | `PUT /users/1` |
| 부분 수정 | `PATCH /users/1` |
| 삭제 | `DELETE /users/1` |

___

## 📌 Route-Controller-Model 패턴

코드를 역할별로 분리해 가독성·재사용성·유지보수성을 높이는 패턴이다. FastAPI에서 주로 사용한다.

계층을 나누는 핵심 이유는 **변경의 격리**다. 예를 들어 DB를 SQLite에서 PostgreSQL로 바꿔도 **Model만 수정하면 되고** Router·Controller는 건드릴 필요가 없다. 각 계층이 자기 역할만 알기 때문에 한 곳의 변경이 다른 곳으로 번지지 않는다.

| 계층 | 역할 |
| --- | --- |
| **Router** | HTTP 요청을 받아 적절한 Controller로 전달 |
| **Controller** | 비즈니스 로직 처리 (입력 검증, 조건 분기) |
| **Model** | 데이터 구조 정의 및 DB 접근 |

**요청 흐름**: `Client → Router → Controller → Model → DB → (역순으로 응답)`

![Route-Controller-Model 요청 흐름](여기에-이미지-드래그)

```
project/
├── main.py
├── routers/        # 요청 라우팅
├── controllers/    # 비즈니스 로직
├── models/         # 데이터 접근
└── dtos/           # 요청/응답 구조 (Pydantic)
```

각 계층은 자기 역할만 담당한다. Router는 요청 전달, Controller는 검증·분기, Model은 데이터 처리를 맡는다.

```python
# routers/user_router.py
router = APIRouter(prefix="/users")

@router.get("/{user_id}")
def get_user(user_id: int):
    return user_controller.get_user(user_id)

# controllers/user_controller.py
def get_user(user_id: int):
    user = user_model.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")
    return {"data": user}
```

___

## 📌 Pydantic — 요청 Body 검증

FastAPI는 **Pydantic** 모델로 요청 데이터의 구조와 타입을 정의한다. 선언한 타입과 다른 값이 오면 <span style="color:red">자동으로 422를 반환</span>한다.

```python
from pydantic import BaseModel

class PostCreate(BaseModel):
    title: str
    content: str

@router.post("", status_code=201)
def create_post(payload: PostCreate):   # 타입 선언만으로 자동 검증
    return post_controller.create_post(payload)
```

___

## 📌 미들웨어와 Exception Handler

- **미들웨어**: 라우터 도달 전 / 응답 나가기 전 **전체 요청에 공통으로 개입**하는 코드다. CORS 처리, Rate Limiting, 요청 시간 측정 등에 사용한다.
- **Exception Handler**: 특정 예외를 전역에서 가로채 **일관된 응답 형식으로 변환**한다. Controller는 `raise`만 하고, 응답 변환은 Handler에서 처리하므로 에러 정책이 한 곳에 모인다.

```python
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code,
                        content={"message": exc.detail, "data": None})
```

___

## 📌 백엔드 개발 흐름

```
1. 필요한 기능 파악
2. API 엔드포인트 설계 (URL, Method, 요청/응답 형태)
3. 프로젝트 구조 잡기
4. 코드 구현 (Router → Controller → Model 순서)
5. 테스트 (Swagger UI: http://localhost:8000/docs)
```

규모가 커지면 Model을 **Service**(비즈니스 규칙)와 **Repository**(DB 접근)로 다시 분리해 역할을 더 명확히 나눌 수 있다.
