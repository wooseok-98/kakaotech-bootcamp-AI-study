# FastAPI 예외 처리 — 쉽게 이해하기

---

## 1. 예외 처리란

코드를 짤 때 우리는 보통 잘 되는 상황만 생각해

근데 실제 서비스에서는 이런 일이 생겨:
- 없는 유저 ID로 조회 요청이 오는 경우
- 숫자를 넣어야 하는 필드에 문자열이 들어오는 경우
- AI 모델 추론 중 메모리가 부족한 경우

이런 예상치 못한 상황을 **예외(Exception)** 라고 해

> **예외 처리 = 문제가 생겼을 때 어떻게 할지 미리 코드로 정해놓는 것**

---

## 2. 예외 처리가 없으면

### 케이스 1 — 200인데 null이 오는 경우

```python
def get_user(user_id: int):
    user = user_model.get_by_id(user_id)
    return {"data": user}   # user가 None이어도 그냥 반환
```

999번 유저가 없으면 `user`가 `None`인데 그냥 return을 해버려. 그러면:

```
GET /users/999
→ 200 OK
→ { "data": null }
```

상태 코드가 200이니까 클라이언트는 성공인 줄 알고 `data`를 사용하려다 또 에러가 남

### 케이스 2 — 500이 뜨는 경우

```python
def predict(payload: dict):
    result = model.predict(payload["text"])   # "text" 키가 없으면?
    return {"result": result}
```

클라이언트가 `text` 없이 요청을 보내면 `KeyError`가 발생하고, 아무도 안 잡으면 FastAPI가 자동으로 500을 반환해

```
POST /predict  (text 없이)
→ 500 Internal Server Error
```

`text` 필드를 빠뜨린 건 클라이언트 실수인데 서버가 500을 내려보내니까 "서버 고장인가?" 하고 자기 코드를 안 고치게 돼

---

## 3. Python 예외 기초

### 예외 계층 구조

```
Exception
├── ValueError     # 잘못된 값  (int("abc"))
├── TypeError      # 잘못된 타입  ("a" + 1)
├── KeyError       # 딕셔너리에 없는 키
├── IndexError     # 리스트 범위 초과
└── ...            # 커스텀 예외도 여기서 상속
```

`ValueError`는 `Exception`을 상속했기 때문에 `ValueError`는 `Exception`의 일종이야. 그래서 `except Exception`이라고 쓰면 `ValueError`도 잡혀

### try / except / finally

```python
try:
    result = model.predict(text)       # 예외가 발생할 수 있는 코드
except ValueError as e:                # ValueError가 발생하면 여기 실행
    print(f"잘못된 값: {e}")
except Exception as e:                 # 그 외 모든 예외
    print(f"알 수 없는 에러: {e}")
finally:
    print("항상 실행")                 # 예외 여부 관계없이 무조건 실행
```

`except`는 위에서부터 순서대로 비교하니까 구체적인 것부터 먼저 써야 해. 순서가 반대면 `Exception`이 전부 다 먼저 잡아버려

### raise

```python
def get_user(user_id: int):
    if user_id <= 0:
        raise ValueError("user_id는 양수여야 합니다")
    ...   # user_id <= 0 이면 여기 실행 안 됨
```

`raise`가 실행되면 그 아래 코드는 실행되지 않고, 예외가 이 함수를 호출한 곳으로 전달돼. 아무도 안 잡으면 계속 위로 올라가다가 FastAPI가 500으로 처리해

---

## 4. HTTPException

`ValueError`는 HTTP 상태 코드를 지정할 방법이 없어. 그래서 FastAPI에서는 상태 코드를 직접 지정할 수 있는 `HTTPException`을 써

```python
from fastapi import HTTPException

def get_user(user_id: int):
    user = user_model.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")
    return {"data": user}
```

`raise HTTPException`을 하는 순간 FastAPI가 즉시 응답을 만들어 반환하고, 그 아래 코드는 실행되지 않아

```json
// 404 응답
{ "detail": "user_not_found" }
```

### 자주 쓰는 status_code

| 상황 | status_code |
| --- | --- |
| 요청 데이터 형식 오류 | 400 |
| 인증되지 않은 사용자 | 401 |
| 권한 없음 | 403 |
| 존재하지 않는 리소스 | 404 |
| 중복 데이터 | 409 |
| 서버 내부 오류 | 500 |
| 서비스 사용 불가 | 503 |

### 한계

```python
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

나중에 메시지를 바꾸려면 모든 파일을 찾아서 수정해야 해 → 커스텀 예외 클래스로 해결

---

## 5. FastAPI 내부 예외 처리 흐름

```
클라이언트 요청
      ↓
  라우터
      ↓
  컨트롤러  ── raise ──→  Exception Handler  ──→ 클라이언트 응답
      ↓                         ↑
    모델     ── raise ──────────┘
```

어디서 예외가 발생하든 Exception Handler가 최종적으로 잡아

FastAPI가 기본으로 등록해두는 Handler 두 개:
- `HTTPException` → `{"detail": "..."}` 형태로 응답
- `RequestValidationError` → 422 응답 (Pydantic 검증 실패 시 자동 발생)

그 외 아무도 안 잡은 예외 → 500

> **예외가 위로 전달된다는 게 뭐야?**
>
> 예외가 발생하면 잡는 코드를 만날 때까지 함수 호출 역순으로 올라가
> ```
> model → controller → router → FastAPI (500 처리)
> ```

---

## 6. Exception Handler

파일마다 응답 형태가 달라지는 걸 막기 위해 `main.py`에 한 번만 등록해두는 것

```python
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

이제 어디서 `raise HTTPException(status_code=404, detail="user_not_found")`를 해도 응답이 항상:

```json
{ "message": "user_not_found", "data": null }
```

예상치 못한 예외도 한 번에 잡으려면:

```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "internal_server_error", "data": None}
    )
```

---

## 7. Pydantic 검증 에러

Pydantic이 요청 데이터 검증에 실패하면 `RequestValidationError`가 발생하고 기본 응답은 422야

```json
{
    "detail": [{ "type": "missing", "loc": ["body", "text"], "msg": "Field required" }]
}
```

이 형태를 우리 서비스 형태로 바꾸려면:

```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    missing_fields = [e["loc"][-1] for e in errors if e["type"] == "missing"]
    # e["loc"] = ["body", "text"] → [-1]로 마지막 값인 "text"만 꺼냄
    # errors 리스트 전체를 순회하기 때문에 누락된 필드가 여러 개여도 전부 잡혀

    return JSONResponse(
        status_code=400,
        content={"message": "invalid_request", "data": {"missing_fields": missing_fields}}
    )
```

---

## 8. 커스텀 예외 클래스

### 왜 쓰는가

HTTPException 반복 작성 문제를 해결하기 위해서야

```python
# 전 — 중복, 메시지 바꾸려면 전부 찾아야 함
raise HTTPException(status_code=404, detail="user_not_found")  # 여러 곳에서 반복

# 후 — 한 줄, exceptions.py 한 곳만 수정하면 됨
raise UserNotFoundException()
```

### 만드는 방법

```python
# exceptions.py

# 기본 클래스 — 모든 커스텀 예외의 부모
class AppException(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)

# 구체적인 예외 클래스
class UserNotFoundException(AppException):
    def __init__(self):
        super().__init__(status_code=404, message="user_not_found")

class ModelNotLoadedException(AppException):
    def __init__(self):
        super().__init__(status_code=503, message="model_not_loaded")
```

### Handler 등록

`AppException` 하나만 등록하면 이걸 상속한 모든 예외를 잡아
(`UserNotFoundException`은 `AppException`의 일종이니까)

```python
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message, "data": None}
    )
```

### exceptions.py와 Handler의 관계

| | 역할 |
| --- | --- |
| `exceptions.py` | 예외 클래스 정의 — "어떤 예외가 있는지, 코드/메시지는 뭔지" |
| `main.py` Handler | 예외를 응답으로 변환 — "발생한 예외를 어떻게 JSON으로 만들지" |

둘 중 하나만 있으면 안 돼
- Handler만 있으면 → Controller에 HTTPException 중복 문제 해결 안 됨
- 커스텀 예외만 있으면 → Handler가 없으니 FastAPI가 500으로 처리해버림

---

## 9. AI 모델 서빙 예외 전략

일반 CRUD API는 "없는 데이터 → 404" 가 거의 전부지만, AI 모델 서빙은 실패 지점이 훨씬 많아

```
1. 입력 데이터 검증 실패   → 400  (클라이언트 잘못)
2. 모델 로드 안 됨         → 503  (서버 준비 중)
3. 모델 추론 실패          → 500  (서버 내부 오류)
4. 추론 시간 초과          → 504  (너무 오래 걸림)
```

```python
async def predict(payload: PredictRequest):
    # 1. 입력 검증
    if not payload.text or not payload.text.strip():
        raise InvalidTextInputException("텍스트가 비어있습니다")

    # 2. 모델 상태 확인
    if not predict_model.is_loaded():
        raise ModelNotLoadedException()

    # 3. 추론 (타임아웃 적용)
    try:
        result = await asyncio.wait_for(
            predict_model.infer(payload.text),
            timeout=10
        )
    except asyncio.TimeoutError:
        raise ModelTimeoutException()    # TimeoutError를 우리 예외로 변환
    except Exception as e:
        raise ModelInferenceException(detail=str(e))

    return {"data": result}
```

`asyncio.TimeoutError`를 그대로 올리면 `AppException` 계열이 아니라서 Handler가 못 잡아. 그래서 우리 예외로 변환해서 던지는 거야

### 서버 시작 시 모델 로드

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    predict_model.load_model()   # 서버 켤 때 딱 한 번만
    yield

app = FastAPI(lifespan=lifespan)
```

모델이 수백 MB라서 요청마다 로드하면 너무 느려. 서버 시작 시 한 번만 메모리에 올려두고 계속 재사용해

---

## 10. 예외 처리 설계 원칙

**4xx vs 5xx 명확히 구분**
- 클라이언트 잘못 → 4xx ("네가 고쳐야 해")
- 서버 내부 오류 → 5xx ("우리가 고쳐야 해")

**에러 메시지에 내부 정보 노출 금지**

```python
# 나쁜 예 — 서버 내부 파일 경로가 노출됨
raise HTTPException(status_code=500, detail=str(e))
# "FileNotFoundError: /server/models/sentiment.pt not found"

# 좋은 예
raise ModelInferenceException()
# "model_inference_failed"
```

**Controller는 raise만, 응답 변환은 Handler에서**

Controller가 응답 형태까지 직접 만들면 파일마다 달라질 수 있어. Controller는 `raise`만 하고 Handler에서 통일해

**예상 가능한 예외만 명시적으로 처리하고, 나머지는 전역 Handler에 위임**