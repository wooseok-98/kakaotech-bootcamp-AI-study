# FastAPI 시작하기
---

## 환경 설정

### 가상환경

프로젝트마다 독립된 파이썬 환경을 만들어 패키지 충돌을 방지

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 의존성 설치

| 패키지 | 설명 |
|--------|------|
| `fastapi` | 파이썬으로 웹 API를 만드는 프레임워크 |
| `uvicorn` | FastAPI 앱을 실행하는 웹 서버 |

```bash
pip install fastapi uvicorn
pip freeze > requirements.txt
```

---

## 구현

### 기본 앱 구성

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
```

- `app = FastAPI()` — 서버 객체 생성
- `@app.get("/")` — `GET /` 요청이 오면 아래 함수 실행
- `return {"Hello": "World"}` — JSON 형태로 응답

### 서버 실행

```bash
uvicorn main:app --reload
```

- `main` — 실행할 파일명 (`main.py`)
- `app` — FastAPI 객체 이름
- `--reload` — 코드 변경 시 서버 자동 재시작 (개발할 때만 사용)

### API 문서 확인

FastAPI는 API 문서를 자동으로 생성

| 주소 | 설명 |
|------|------|
| `http://localhost:8000/docs` | Swagger UI — 브라우저에서 API 직접 테스트 가능 |
| `http://localhost:8000/redoc` | ReDoc — 읽기 전용 문서 |