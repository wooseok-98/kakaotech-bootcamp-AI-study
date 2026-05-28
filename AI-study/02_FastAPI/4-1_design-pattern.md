# 4-1. 디자인 패턴

자주 발생하는 설계 문제를 해결하는 정형화된 설계 템플릿  
코드를 역할별로 분리해 가독성, 재사용성, 유지보수성을 높이는 것이 목적

---

## Route-Controller-Model 패턴

FastAPI에서 주로 사용하는 아키텍처 패턴

| 계층 | 역할 |
| --- | --- |
| Router | HTTP 요청을 받아 적절한 Controller로 전달 |
| Controller | 비즈니스 로직 처리 (입력 검증, 조건 분기 등) |
| Model | 데이터 구조 정의 및 DB 접근 |

**요청 흐름**: `Client → Router → Controller → Model → DB → (역순으로 응답)`

---

## 디렉토리 구조

```
project/
├── main.py
├── routers/
│   └── user_router.py
├── controllers/
│   └── user_controller.py
├── models/
│   └── user_model.py
└── dtos/
    └── user_dto.py    ← 요청/응답 데이터 구조 정의 (Pydantic)
```

---

## 코드 예시

### main.py

```python
from fastapi import FastAPI
from routers.user_router import router as user_router

app = FastAPI()
app.include_router(user_router, tags=["users"])
```

### routers/user_router.py

```python
from fastapi import APIRouter
from controllers import user_controller

router = APIRouter(prefix="/users")

@router.get("")
def get_users():
    return user_controller.get_users()

@router.get("/{user_id}")
def get_user(user_id: int):
    return user_controller.get_user(user_id)

@router.post("", status_code=201)
def create_user(data: dict):
    return user_controller.create_user(data)
```

### controllers/user_controller.py

```python
from fastapi import HTTPException
from models import user_model

def get_user(user_id: int):
    if user_id <= 0:
        raise HTTPException(status_code=400, detail="invalid_user_id")

    user = user_model.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")

    return {"data": user}

def create_user(data: dict):
    name = data.get("name")
    email = data.get("email")

    if not name or not email:
        raise HTTPException(status_code=400, detail="missing_required_fields")

    if user_model.get_user_by_email(email):
        raise HTTPException(status_code=409, detail="email_already_exists")

    new_user = {"id": len(user_model.get_users()) + 1, "name": name, "email": email}
    user_model.add_user(new_user)
    return {"data": new_user}
```

### models/user_model.py

```python
_users = [
    {"id": 1, "name": "Alice", "email": "alice@test.com"},
    {"id": 2, "name": "Bob", "email": "bob@test.com"},
]

def get_users():
    return _users.copy()

def get_user_by_id(user_id: int):
    return next((u for u in _users if u["id"] == user_id), None)

def get_user_by_email(email: str):
    return next((u for u in _users if u["email"] == email), None)

def add_user(user: dict):
    _users.append(user)
    return user
```

---

## Pydantic — 요청 Body 정의

FastAPI에서 요청 데이터의 구조와 타입을 정의하는 방법  
선언한 타입과 다른 값이 오면 **자동으로 422 반환**

```python
# dtos/post_dto.py
from pydantic import BaseModel

class PostCreate(BaseModel):
    title: str
    content: str

class PostUpdate(BaseModel):
    title: str
    content: str
```

Router에서 사용:

```python
@router.post("", status_code=201)
def create_post(payload: PostCreate):   # 타입 선언만으로 자동 검증
    return post_controller.create_post(payload)
```

---

## Service-Repository 분리 (심화)

규모가 커지면 Model을 다시 역할별로 분리

| 계층 | 역할 |
| --- | --- |
| Service | 비즈니스 규칙 처리 (암호화, 중복 검사, 흐름 제어 등) |
| Repository | DB 접근 전담 (SELECT, INSERT, UPDATE, DELETE) |

```
project/
├── controllers/
├── services/
├── repositories/
└── models/      ← 데이터 구조(스키마) 정의만 담당
```

**흐름**: `Controller → Service → Repository → DB`

Controller는 요청/응답만, Service는 규칙만, Repository는 쿼리만 담당 → 역할이 명확해져 변경이 쉬워짐

---

## 미들웨어

라우터 도달 전 / 응답 나가기 전 **전체 요청에 공통으로 개입**하는 코드

```python
@app.middleware("http")
async def process_time_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

주요 사용처: CORS 처리, Rate Limiting, 요청 처리 시간 측정

---

## Exception Handler

특정 예외를 전역에서 가로채 **일관된 응답 형식으로 변환**

```python
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"message": "INVALID_REQUEST", "data": None}
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail, "data": None}
    )
```

Controller에서는 `raise`만 하고, 응답 변환은 Handler에서 처리 → 에러 정책이 한 곳에 모임

---

## 백엔드 개발 흐름

```
1. 어떤 기능이 필요한지 파악
2. API 엔드포인트 설계 (URL, Method, 요청/응답 형태)
3. 프로젝트 구조 잡기
4. 코드 구현 (Router → Controller → Model 순서)
5. 테스트 (Swagger UI: http://localhost:8000/docs)
```

### REST API 설계 방법

**1단계: 리소스 파악** — 서비스에서 다루는 데이터 정의 (예: User, Post)

**2단계: 엔드포인트 설계** — HTTP 메서드 + URL

```
GET    /posts           → 목록 조회
GET    /posts/{id}      → 상세 조회
POST   /posts           → 생성
PUT    /posts/{id}      → 전체 수정
PATCH  /posts/{id}      → 부분 수정
DELETE /posts/{id}      → 삭제
```

**3단계: 요청/응답 형태 정의**

```
POST /posts
  요청: { "title": "제목", "content": "내용" }
  응답: { "data": { "id": 1, "title": "제목", "content": "내용" } }
```

---

## 자주 쓰는 패턴 요약

```python
# Path Variable
@router.get("/{id}")
def get(id: int): ...

# Query String
@router.get("")
def get_list(page: int = 1, size: int = 10): ...

# Request Body
@router.post("")
def create(payload: SomeDto): ...

# 에러 반환
raise HTTPException(status_code=404, detail="not_found")
```
