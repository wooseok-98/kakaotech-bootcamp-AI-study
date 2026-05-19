# FastAPI

Python으로 웹 API 서버를 만드는 웹 프레임워크
ASGI 기반으로 비동기 처리를 지원하며, 타입 힌트를 활용한 자동 문서화가 특징

---

## 동기 vs 비동기

| 구분 | 설명 |
| --- | --- |
| 동기 (Sync) | 한 작업이 끝나야 다음 작업 시작, 요청 처리 중 다른 요청은 대기 |
| 비동기 (Async) | 작업을 기다리는 동안 다른 작업 처리 가능, 동시 요청 처리에 유리 |

- **WSGI**: 동기 표준 인터페이스 (Flask, Django에서 사용)
- **ASGI**: 비동기 표준 인터페이스 (FastAPI, Starlette에서 사용)

---

## 설치

```bash
pip install fastapi
pip install "uvicorn[standard]"
```

> **Uvicorn**: 
- FastAPI 앱을 실제로 구동하는 ASGI 서버
- FastAPI 자체는 실행 엔진이 없어 Uvicorn이 필요.  

---

## 기본 앱 구성

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello!"}
```

### 서버 실행

```bash
uvicorn main:app --reload
```

| 옵션 | 설명 |
| --- | --- |
| `--reload` | 코드 변경 시 서버 자동 재시작 (개발용) |
| `--host 0.0.0.0` | 외부 접속 허용 |
| `--port 8080` | 포트 지정 |

실행 후 `http://localhost:8000` 으로 확인.

---

## FastAPI vs Django vs Flask

| 구분 | FastAPI | Django | Flask |
| --- | --- | --- | --- |
| 종류 | 마이크로 프레임워크 | 풀스택 프레임워크 | 마이크로 프레임워크 |
| 성능 | 매우 빠름 (비동기) | 보통 (WSGI) | 단순 처리는 빠름 |
| 특징 | 비동기, 자동 문서화, 타입 검증 | ORM·인증 등 내장 기능 풍부 | 자유도 높음, 최소 기능 |
| 적합한 규모 | 소규모~대규모 API | 대규모 서비스 | 소규모, 프로토타입 |

---

# pyproject.toml

Python 프로젝트의 설정과 의존성을 관리하는 표준 설정 파일

---

## 구조

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.27",
]

[project.optional-dependencies]
dev = ["pytest", "httpx", "ruff"]
```

| 섹션 | 설명 |
| --- | --- |
| `[project]` | 프로젝트 이름, 버전, Python 버전, 의존성 정의 |
| `[project.optional-dependencies]` | 개발 환경에서만 필요한 패키지 그룹 |

### 의존성 설치

```bash
pip install .          # dependencies 설치
pip install .[dev]     # optional-dependencies.dev 까지 설치
```

---

## 참고: pip / venv

```bash
# 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # macOS/Linux
```

- **pip**: Python 패키지 관리자. PyPI에서 패키지를 내려받아 설치
- **venv**: 프로젝트별 독립 실행환경. 패키지 버전 충돌 방지

---

## 참고: 설치된 패키지 목록 확인

```bash
pip freeze > requirements.txt
```

---

# Gunicorn

Python 웹 앱을 위한 WSGI 서버. 운영 환경에서 **프로세스 관리자** 역할.  
Uvicorn 단독 실행은 개발용에 적합하고, 운영 환경에서는 Gunicorn과 조합해 사용.

> **Gunicorn + Uvicorn 구조**:  
> Gunicorn(마스터)이 여러 Uvicorn 워커를 관리 → 장애 발생 시 자동 복구, CPU 코어 병렬 활용

---

## 설치

```bash
pip install gunicorn
gunicorn -v  # 설치 확인
```

> Windows에서는 Gunicorn 실행 불가. 로컬 개발은 Uvicorn으로 진행.

---

## 서버 실행

```bash
gunicorn -k uvicorn.workers.UvicornWorker main:app -w 4 -b 0.0.0.0:8000
```

| 옵션 | 설명 |
| --- | --- |
| `-k uvicorn.workers.UvicornWorker` | Uvicorn을 워커로 사용 |
| `main:app` | `main.py`의 `app` 객체 지정 |
| `-w 4` | 워커 수 (CPU 코어 수에 맞게 설정) |
| `-b 0.0.0.0:8000` | 바인딩 주소:포트 |

종료: `Ctrl + C`
