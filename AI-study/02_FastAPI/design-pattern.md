# 디자인 패턴

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
└── models/
    └── user_model.py
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

## 참고: 미들웨어

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

## 참고: Exception Handler

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