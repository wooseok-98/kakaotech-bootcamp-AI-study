# FastAPI 예외 처리 딥다이브

FastAPI에서 기본 예외 처리와 커스텀 예외 클래스를 활용하여 API 안정성을 높이는 방법과,
AI 모델 예측 실패 및 잘못된 입력 데이터에 대한 구체적인 대응 전략

---

## 목차

1. [예외 처리란 무엇인가](#1-예외-처리란-무엇인가)
2. [예외 처리가 없으면 어떻게 되는가](#2-예외-처리가-없으면-어떻게-되는가)
3. [Python 예외 기초](#3-python-예외-기초)
4. [FastAPI 기본 예외 처리 — HTTPException](#4-fastapi-기본-예외-처리--httpexception)
5. [FastAPI 내부에서 예외가 처리되는 흐름](#5-fastapi-내부에서-예외가-처리되는-흐름)
6. [Exception Handler — 전역 예외 처리](#6-exception-handler--전역-예외-처리)
7. [Pydantic 검증 에러 처리](#7-pydantic-검증-에러-처리)
8. [커스텀 예외 클래스 설계](#8-커스텀-예외-클래스-설계)
9. [AI 모델 서빙 예외 전략](#9-ai-모델-서빙-예외-전략)
10. [실전 코드 — 감성 분석 API](#10-실전-코드--감성-분석-api)
11. [예외 처리 설계 원칙](#11-예외-처리-설계-원칙)

---

## 1. 예외 처리란 무엇인가

프로그램이 실행되는 동안 예상치 못한 상황이 발생했을 때 이를 감지하고 적절하게 처리하는 것

예상치 못한 상황의 예:
- 없는 사용자 ID로 조회 요청
- 숫자여야 하는 필드에 문자열이 들어옴
- AI 모델 추론 중 메모리 부족
- 외부 API 호출 타임아웃

예외 처리를 하지 않으면 서버가 500 에러를 내뱉거나 아예 죽어버림.
예외 처리를 잘 하면 **무슨 문제가 발생했는지 명확하게 클라이언트에 알려줄 수 있음**

---

## 2. 예외 처리가 없으면 어떻게 되는가

아래처럼 예외 처리가 전혀 없는 API가 있다고 가정

```python
@router.get("/{user_id}")
def get_user(user_id: int):
    user = user_model.get_by_id(user_id)
    return {"data": user}  # user가 None이어도 그냥 반환
```

존재하지 않는 ID로 요청하면:

```json
// GET /users/999
{
    "data": null
}
```

`return`을 정상적으로 실행했기 때문에 상태 코드가 200으로 내려옴.
클라이언트 입장에서는 성공인지 에러인지 판단 불가

더 심각한 경우:

```python
@router.post("/predict")
def predict(payload: dict):
    result = model.predict(payload["text"])  # text 키가 없으면?
    return {"result": result}
```

`payload["text"]`에서 `KeyError` 발생 → FastAPI가 500(서버 내부 오류) 반환 → 클라이언트는 원인을 알 수 없음

---

## 3. Python 예외 기초

### 예외 계층 구조

```
BaseException
└── Exception
    ├── ValueError       # 잘못된 값
    ├── TypeError        # 잘못된 타입
    ├── KeyError         # 딕셔너리에 없는 키
    ├── IndexError       # 리스트 범위 초과
    ├── FileNotFoundError
    └── ...              # 커스텀 예외도 여기서 상속
```

### try / except / finally

```python
try:
    result = model.predict(text)       # 예외가 발생할 수 있는 코드
except ValueError as e:
    print(f"잘못된 값: {e}")           # ValueError 발생 시
except Exception as e:
    print(f"알 수 없는 에러: {e}")     # 그 외 모든 예외
finally:
    print("항상 실행")                 # 예외 여부와 관계없이 실행
```

### raise — 예외를 직접 발생시키기

```python
def get_user(user_id: int):
    if user_id <= 0:
        raise ValueError("user_id는 양수여야 합니다")
    ...
```

---

## 4. FastAPI 기본 예외 처리 — HTTPException

### 기본 사용법

```python
from fastapi import HTTPException

def get_user(user_id: int):
    user = user_model.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")
    return {"data": user}
```

`raise HTTPException`을 하면 FastAPI가 즉시 해당 상태 코드와 메시지로 응답을 만들어 반환.
그 아래 코드는 실행되지 않음

응답:
```json
// 404
{
    "detail": "user_not_found"
}
```

### status_code 선택 기준

| 상황 | status_code |
| --- | --- |
| 요청 데이터 형식 오류 | 400 Bad Request |
| 인증되지 않은 사용자 | 401 Unauthorized |
| 권한 없음 | 403 Forbidden |
| 존재하지 않는 리소스 | 404 Not Found |
| 중복 데이터 | 409 Conflict |
| 서버 내부 오류 | 500 Internal Server Error |
| 서비스 사용 불가 | 503 Service Unavailable |

### headers 파라미터

```python
raise HTTPException(
    status_code=401,
    detail="not_authenticated",
    headers={"WWW-Authenticate": "Bearer"},  # 인증 방식 안내
)
```

### HTTPException의 한계

```python
# 여러 곳에서 같은 에러 코드를 반복 작성
def get_user(user_id):
    user = user_model.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")

def update_user(user_id, payload):
    user = user_model.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")  # 중복

def delete_user(user_id):
    user = user_model.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")  # 중복
```

- 같은 에러를 여러 곳에서 반복 작성 → 나중에 메시지 바꾸려면 모두 찾아서 수정해야 함
- 어떤 에러인지 코드만 봐서는 의미를 파악하기 어려움
- 에러 응답 형태를 일관되게 유지하기 어려움

---

## 5. FastAPI 내부에서 예외가 처리되는 흐름

```
클라이언트 요청
      ↓
  미들웨어
      ↓
  라우터
      ↓
  컨트롤러  ── raise HTTPException ──→  Exception Handler  ──→ 클라이언트 응답
      ↓                                       ↑
    모델        ── raise Exception ───────────┘
```

FastAPI는 앱 시작 시 기본 Exception Handler 두 개를 자동 등록함:

1. `HTTPException` → `http_exception_handler` — `{"detail": "..."}` 형태로 응답
2. `RequestValidationError` → `request_validation_exception_handler` — 422 응답
   - Pydantic이 검사에서 실패했을 때 자동으로 발생시키는 예외 (필수 필드를 빠뜨린 경우 등)

그 외 처리되지 않은 예외 → 500 Internal Server Error

`@app.exception_handler()`로 이 기본 핸들러를 덮어쓰거나 새로운 예외 타입을 등록할 수 있음

---

## 6. Exception Handler — 전역 예외 처리

파일마다 응답 형태가 달라지는 것을 방지하기 위해 `main.py`에 한 번만 등록하여 모든 예외를 같은 형태로 응답

### 기본 구조

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(예외_클래스)
async def 핸들러_이름(request: Request, exc: 예외_클래스):
    return JSONResponse(
        status_code=상태코드,
        content={"message": "...", "data": None}
    )
```

### HTTPException 핸들러 커스터마이징

기본 HTTPException 응답은 `{"detail": "..."}` 형태.
응답 형태를 `{"message": "...", "data": None}`으로 통일하고 싶다면:

```python
from fastapi.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.detail,
            "data": None
        }
    )
```

### 처리되지 않은 모든 예외 잡기

```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "message": "internal_server_error",
            "data": None
        }
    )
```

이렇게 하면 개발자가 예상하지 못한 예외가 발생해도 500 에러가 그대로 노출되지 않음

---

## 7. Pydantic 검증 에러 처리

Pydantic이 요청 데이터를 검증할 때 실패하면 `RequestValidationError` 발생 → 기본 응답은 422

```python
# dtos/predict_dto.py
from pydantic import BaseModel

class PredictRequest(BaseModel):
    text: str
    max_length: int = 512
```

`text` 필드 없이 요청하면 기본 422 응답:
```json
{
    "detail": [
        {
            "type": "missing",
            "loc": ["body", "text"],
            "msg": "Field required",
            ...
        }
    ]
}
```

이 응답 형태를 일관되게 바꾸려면:

```python
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    missing_fields = [e["loc"][-1] for e in errors if e["type"] == "missing"]

    return JSONResponse(
        status_code=400,
        content={
            "message": "invalid_request",
            "data": {
                "missing_fields": missing_fields,
                "detail": errors
            }
        }
    )
```

---

## 8. 커스텀 예외 클래스 설계

### 왜 필요한가

4번에서 봤던 HTTPException의 한계:

```python
# 같은 에러를 여러 곳에서 반복 작성
def get_user(user_id):
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")

def update_user(user_id, payload):
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")  # 중복

def delete_user(user_id):
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")  # 중복
```

커스텀 예외 클래스를 쓰면:

```python
raise UserNotFoundException()   # 세 곳 모두 이 한 줄로 대체
```

- **중복 제거** — 메시지 바꾸려면 `exceptions.py` 한 곳만 수정
- **가독성** — 클래스 이름만 봐도 어떤 상황인지 바로 파악

---

### 기본 커스텀 예외

`Exception`을 상속해서 `status_code`와 `message`를 담을 수 있는 기본 예외 클래스를 만듦

```python
# exceptions.py
class AppException(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)
```

### 도메인별 예외 클래스

`AppException`을 상속해서 구체적인 예외 클래스를 만듦

```python
# 사용자 관련
class UserNotFoundException(AppException):
    def __init__(self):
        super().__init__(status_code=404, message="user_not_found")

class EmailAlreadyExistsException(AppException):
    def __init__(self):
        super().__init__(status_code=409, message="email_already_exists")

# AI 모델 관련
class ModelNotLoadedException(AppException):
    def __init__(self):
        super().__init__(status_code=503, message="model_not_loaded")

class ModelInferenceException(AppException):
    def __init__(self, detail: str = ""):
        super().__init__(status_code=500, message=f"model_inference_failed: {detail}")

class InvalidInputException(AppException):
    def __init__(self, detail: str = ""):
        super().__init__(status_code=400, message=f"invalid_input: {detail}")
```

### Exception Handler 등록

```python
# main.py
from exceptions import AppException

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.message,
            "data": None
        }
    )
```

### 사용 예시 — 코드가 얼마나 깔끔해지는가

```python
# 전: HTTPException 직접 사용
def get_user(user_id: int):
    user = user_model.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")

# 후: 커스텀 예외 사용
def get_user(user_id: int):
    user = user_model.get_by_id(user_id)
    if not user:
        raise UserNotFoundException()
```

- 상태 코드와 메시지가 예외 클래스 안에 캡슐화됨
- 나중에 메시지를 바꾸려면 `exceptions.py` 한 곳만 수정
- 코드 읽을 때 어떤 에러인지 클래스 이름만 봐도 바로 파악 가능

### 커스텀 예외 클래스와 Exception Handler의 관계

둘은 짝꿍이야. 하나만 있으면 완성이 안 됨

| | 역할 |
| --- | --- |
| 커스텀 예외 클래스 | "무슨 문제인지" 이름 붙이기 + 코드/메시지 한 곳에 모으기 |
| Exception Handler | 발생한 예외를 받아서 응답 형태로 변환하기 |
| Controller | raise만 하면 끝 |

```
커스텀 예외 없이 Handler만 있으면
→ Controller 코드에 HTTPException 중복 작성 문제 해결 안 됨

Handler 없이 커스텀 예외만 있으면
→ FastAPI가 모르는 예외로 취급해서 그냥 500으로 처리해버림

둘 다 있으면
→ Controller는 raise UserNotFoundException() 한 줄
→ Handler가 받아서 {"message": "user_not_found", "data": null} 로 변환
```

---

## 9. AI 모델 서빙 예외 전략

AI 모델 서빙 API는 일반 CRUD API와 달리 다양한 실패 지점이 있음

### 발생 가능한 예외 시나리오

```
클라이언트 요청
      ↓
1. 입력 데이터 검증 실패      → 400 (클라이언트 잘못)
      ↓
2. 모델 로딩 상태 확인        → 503 (서버 준비 안 됨)
      ↓
3. 전처리 실패               → 400 or 500
      ↓
4. 모델 추론 실패             → 500
      ↓
5. 추론 시간 초과             → 504
      ↓
응답 반환
```

### 예외 클래스 설계

```python
# exceptions.py

class ModelNotLoadedException(AppException):
    """모델이 아직 로드되지 않았거나 로드에 실패한 경우"""
    def __init__(self):
        super().__init__(status_code=503, message="model_not_loaded")

class ModelInferenceException(AppException):
    """모델 추론 중 오류 발생"""
    def __init__(self, detail: str = ""):
        super().__init__(status_code=500, message="model_inference_failed")
        self.detail = detail

class InvalidTextInputException(AppException):
    """텍스트 입력이 유효하지 않은 경우 (빈 문자열, 너무 긴 텍스트 등)"""
    def __init__(self, detail: str):
        super().__init__(status_code=400, message=f"invalid_text_input: {detail}")

class ModelTimeoutException(AppException):
    """모델 추론이 제한 시간을 초과한 경우"""
    def __init__(self):
        super().__init__(status_code=504, message="model_inference_timeout")
```

### Controller에서 예외 처리

```python
# controllers/predict_controller.py
import asyncio
from exceptions import (
    ModelNotLoadedException,
    ModelInferenceException,
    InvalidTextInputException,
    ModelTimeoutException
)
from models import predict_model

INFERENCE_TIMEOUT = 10  # 10초 제한 (추후 한 곳에서 바꾸기 위해 상수로 선언)

def validate_text(text: str):
    if not text or not text.strip():
        raise InvalidTextInputException("텍스트가 비어있습니다")
    if len(text) > 5000:
        raise InvalidTextInputException(f"텍스트가 너무 깁니다 (최대 5000자, 현재 {len(text)}자)")

async def predict(text: str):
    # 1. 입력 검증
    validate_text(text)

    # 2. 모델 로드 상태 확인
    if not predict_model.is_loaded():
        raise ModelNotLoadedException()

    # 3. 추론 (타임아웃 적용)
    try:
        result = await asyncio.wait_for(
            predict_model.infer(text),
            timeout=INFERENCE_TIMEOUT
        )
    except asyncio.TimeoutError:
        raise ModelTimeoutException()
    except Exception as e:
        raise ModelInferenceException(detail=str(e))

    return {"data": result}
```

### Model에서 예외 처리

```python
# models/predict_model.py
_model = None

def load_model():
    global _model
    try:
        # 실제로는 여기서 torch, transformers 등으로 모델 로드
        _model = {"loaded": True}  # 예시
        print("모델 로드 완료")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        _model = None

def is_loaded() -> bool:
    return _model is not None

async def infer(text: str) -> dict:
    if _model is None:
        raise RuntimeError("모델이 로드되지 않았습니다")
    try:
        # 실제로는 모델 추론 결과 반환
        return {"label": "positive", "score": 0.95}
    except Exception as e:
        raise RuntimeError(f"추론 실패: {e}")
```

### 서버 시작 시 모델 로드

```python
# main.py
from contextlib import asynccontextmanager
from models import predict_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    predict_model.load_model()  # 서버 시작 시 모델 로드
    yield
    # 서버 종료 시 정리 작업 (필요하면)

app = FastAPI(lifespan=lifespan)
```

---

## 10. 실전 코드 — 감성 분석 API

텍스트를 보내면 긍정/부정을 분석해주는 API 예시. 1~9번의 내용을 전부 합친 완성 코드

전체 프로젝트 구조:

```
sentiment-api/
├── main.py                     ← 앱 생성 + Handler 등록 + 라우터 등록
├── exceptions.py               ← 커스텀 예외 클래스 모음
├── dtos/
│   └── predict_dto.py          ← 요청 데이터 형태 정의
├── routers/
│   └── predict_router.py       ← URL 경로 정의
├── controllers/
│   └── predict_controller.py   ← 입력 검증 + 추론 요청 + 예외 처리
└── models/
    └── predict_model.py        ← 실제 AI 추론
```

### exceptions.py

서비스에서 발생할 수 있는 예외들을 클래스로 정의해두는 파일

```python
class AppException(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)

class ModelNotLoadedException(AppException):
    def __init__(self):
        super().__init__(503, "model_not_loaded")

class ModelInferenceException(AppException):
    def __init__(self, detail: str = ""):
        super().__init__(500, "model_inference_failed")
        self.detail = detail

class InvalidTextInputException(AppException):
    def __init__(self, detail: str):
        super().__init__(400, f"invalid_text_input: {detail}")

class ModelTimeoutException(AppException):
    def __init__(self):
        super().__init__(504, "model_inference_timeout")
```

### dtos/predict_dto.py

클라이언트가 보내는 요청 데이터의 형태와 타입을 정의하는 파일

```python
from pydantic import BaseModel

class PredictRequest(BaseModel):
    text: str
    max_length: int = 512
```

### models/predict_model.py

실제 AI 모델을 로드하고 추론을 실행하는 파일

```python
_model = None

def load_model():
    global _model
    try:
        _model = {"name": "sentiment-model", "loaded": True}
        print("모델 로드 완료")
    except Exception as e:
        print(f"모델 로드 실패: {e}")

def is_loaded() -> bool:
    return _model is not None

async def infer(text: str) -> dict:
    # 실제로는 모델 추론 로직
    return {"label": "positive", "score": 0.92}
```

### controllers/predict_controller.py

입력 검증, 모델 상태 확인, 추론 요청, 예외 처리 등 비즈니스 로직을 담당하는 파일

```python
import asyncio
from exceptions import (
    ModelNotLoadedException, ModelInferenceException,
    InvalidTextInputException, ModelTimeoutException
)
from models import predict_model
from dtos.predict_dto import PredictRequest

INFERENCE_TIMEOUT = 10

def _validate_text(text: str):
    if not text or not text.strip():
        raise InvalidTextInputException("텍스트가 비어있습니다")
    if len(text) > 5000:
        raise InvalidTextInputException(f"최대 5000자 (현재 {len(text)}자)")

async def predict(payload: PredictRequest):
    _validate_text(payload.text)

    if not predict_model.is_loaded():
        raise ModelNotLoadedException()

    try:
        result = await asyncio.wait_for(
            predict_model.infer(payload.text),
            timeout=INFERENCE_TIMEOUT
        )
    except asyncio.TimeoutError:
        raise ModelTimeoutException()
    except Exception as e:
        raise ModelInferenceException(detail=str(e))

    return {"data": result}
```

### routers/predict_router.py

URL 경로와 Controller 함수를 연결하는 파일

```python
from fastapi import APIRouter
from controllers import predict_controller
from dtos.predict_dto import PredictRequest

router = APIRouter()

@router.post("")
async def predict(payload: PredictRequest):
    return await predict_controller.predict(payload)
```

### main.py

앱 생성, Exception Handler 등록, 라우터 등록을 하는 진입점 파일

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from exceptions import AppException
from models import predict_model
from routers.predict_router import router as predict_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    predict_model.load_model()
    yield

app = FastAPI(lifespan=lifespan)

# 커스텀 예외 핸들러
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message, "data": None}
    )

# Pydantic 검증 에러 핸들러
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"message": "invalid_request", "data": None}
    )

# 예상치 못한 예외 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "internal_server_error", "data": None}
    )

app.include_router(predict_router, prefix="/predict", tags=["predict"])
```

### 각 에러 케이스별 응답 예시

```
POST /predict  { "text": "" }
→ 400  { "message": "invalid_text_input: 텍스트가 비어있습니다", "data": null }

POST /predict  { "text": "이 영화 정말 재밌다" }  (모델 로드 안 됨)
→ 503  { "message": "model_not_loaded", "data": null }

POST /predict  (text 필드 없음)
→ 400  { "message": "invalid_request", "data": null }

POST /predict  { "text": "이 영화 정말 재밌다" }  (정상)
→ 200  { "data": { "label": "positive", "score": 0.92 } }
```

---

## 11. 예외 처리 설계 원칙

### 1. 클라이언트 잘못 vs 서버 잘못을 명확히 구분

| 상황 | 상태 코드 | 의미 |
| --- | --- | --- |
| 잘못된 입력 데이터 | 4xx | 클라이언트가 고쳐야 함 |
| 서버/모델 내부 오류 | 5xx | 서버 측 문제 |

클라이언트 잘못인데 500을 반환하면 → 클라이언트는 자기 코드 고쳐야 하는 걸 모름

### 2. 에러 메시지에 민감한 정보를 포함하지 않는다

```python
# 나쁜 예 — 내부 경로, 스택 트레이스 노출
raise HTTPException(status_code=500, detail=str(e))

# 좋은 예 — 일반적인 메시지만
raise ModelInferenceException()
# 내부 로그에는 상세 기록
```

### 3. 예외는 발생한 곳에서 바로 raise, 응답 변환은 Handler에서

응답 형태가 달라지는 것을 방지하기 위해 Controller는 raise만 하고, 응답 형태는 Handler에서 통일

```python
# Controller — raise만 하고 응답 형태 신경 쓰지 않음
if not user:
    raise UserNotFoundException()

# Handler — 응답 형태만 담당
@app.exception_handler(AppException)
async def handler(request, exc):
    return JSONResponse(...)
```

### 4. 모든 예외를 잡으려 하지 않는다

예상 가능한 예외는 명시적으로 처리하고, 나머지는 전역 핸들러에 위임

```python
try:
    result = model.infer(text)
except MemoryError:
    raise ModelInferenceException("메모리 부족")
except TimeoutError:
    raise ModelTimeoutException()
# 그 외는 Exception 핸들러가 처리
```

### 5. 예외 클래스는 도메인별로 분리

```
exceptions/
├── __init__.py
├── base.py          # AppException
├── user.py          # UserNotFoundException, EmailAlreadyExistsException
└── model.py         # ModelNotLoadedException, ModelInferenceException
```