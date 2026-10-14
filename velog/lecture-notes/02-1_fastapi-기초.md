<!-- 📝 글 소개 (출간 시 입력): FastAPI 개념과 동기/비동기, 설치·실행, 프레임워크 비교 정리 -->

# 🎈 FastAPI 시작하기

**FastAPI**는 Python으로 웹 API 서버를 만드는 웹 프레임워크다. <span style="color:red">ASGI</span> 기반으로 비동기 처리를 지원하며, 타입 힌트를 활용한 자동 문서화가 특징이다.

타입 힌트를 그대로 활용해 **요청 데이터 검증(Pydantic)과 API 문서가 자동 생성**된다. 서버 실행 후 `/docs`에서 <span style="color:red">Swagger UI</span>, `/redoc`에서 ReDoc 형식의 문서를 바로 확인할 수 있다.

___

## 📌 동기 vs 비동기

**동기**는 한 작업의 결과를 기다리는 동안 다음 작업을 시작하지 못하고, **비동기**는 기다리는 동안 다른 작업을 처리한다. DB 조회나 외부 API 호출처럼 **I/O 대기 시간**이 있을 때 비동기가 유리하다.

FastAPI에서는 경로 함수를 정의하는 방식에 따라 처리가 달라진다.

- `async def`로 정의하면 **비동기**로 처리되고, `await`로 I/O를 기다리는 동안 이벤트 루프가 다른 요청을 처리한다
- 일반 `def`로 정의하면 FastAPI가 **별도 스레드풀**에서 실행해 이벤트 루프를 막지 않는다

> 주의: `async def` 함수 안에서 `time.sleep()`이나 동기 DB 드라이버 같은 **블로킹 코드를 호출하면 이벤트 루프 전체가 멈춘다**. 비동기 함수 안에서는 `asyncio.sleep()`처럼 비동기를 지원하는 라이브러리를 사용해야 한다.

| 구분 | 설명 |
| --- | --- |
| **동기 (Sync)** | 한 작업이 끝나야 다음 작업을 시작한다. 요청 처리 중 다른 요청은 대기한다 |
| **비동기 (Async)** | 작업을 기다리는 동안 다른 작업을 처리한다. 동시 요청 처리에 유리하다 |

- **WSGI**: 동기 표준 인터페이스 (Flask, Django에서 사용)
- **ASGI**: 비동기 표준 인터페이스 (FastAPI, Starlette에서 사용)

![동기 vs 비동기 처리 비교](여기에-이미지-드래그)

___

## 📌 설치와 기본 앱

FastAPI는 실행 엔진이 없어 ASGI 서버인 **Uvicorn**이 함께 필요하다.

```bash
pip install fastapi
pip install "uvicorn[standard]"
```

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello!"}
```

```bash
uvicorn main:app --reload   # http://localhost:8000
```

| 옵션 | 설명 |
| --- | --- |
| `--reload` | 코드 변경 시 서버 자동 재시작 (개발용) |
| `--host 0.0.0.0` | 외부 접속 허용 |
| `--port 8080` | 포트 지정 |

___

## 📌 FastAPI vs Django vs Flask

세 프레임워크는 **목적이 다르다**. Django는 ORM·인증·관리자 페이지까지 내장한 풀스택이라 큰 서비스에 적합하고, Flask는 최소 기능만 제공해 자유도가 높다. **FastAPI는 비동기와 타입 검증에 강해 API 서버, 특히 AI 모델 서빙에 자주 선택된다.**

| 구분 | **FastAPI** | Django | Flask |
| --- | --- | --- | --- |
| 종류 | 마이크로 | 풀스택 | 마이크로 |
| 성능 | 매우 빠름 (비동기) | 보통 (WSGI) | 단순 처리는 빠름 |
| 특징 | 비동기·자동 문서화·타입 검증 | ORM·인증 등 내장 | 자유도 높음, 최소 기능 |
| 적합 규모 | 소~대규모 API | 대규모 서비스 | 소규모, 프로토타입 |

___

## 📌 패키지 관리

`pyproject.toml`은 Python 프로젝트의 설정과 의존성을 관리하는 표준 설정 파일이다.

```toml
[project]
name = "my-project"
requires-python = ">=3.11"
dependencies = ["fastapi>=0.115", "uvicorn[standard]>=0.27"]

[project.optional-dependencies]
dev = ["pytest", "httpx", "ruff"]
```

```bash
pip install .          # dependencies 설치
pip install .[dev]     # dev 의존성까지 설치
```

- **venv**: 프로젝트별 독립 실행환경으로 패키지 버전 충돌을 방지한다
- **pip freeze > requirements.txt**: 설치된 패키지 목록을 저장한다

___

## 📌 운영 환경: Gunicorn

Uvicorn 단독 실행은 개발용에 적합하고, 운영 환경에서는 **Gunicorn**과 조합해 사용한다. Gunicorn(마스터)이 여러 Uvicorn 워커를 관리해 장애 발생 시 자동 복구하고 CPU 코어를 병렬로 활용한다.

```bash
gunicorn -k uvicorn.workers.UvicornWorker main:app -w 4 -b 0.0.0.0:8000
```

> Windows에서는 Gunicorn 실행이 불가능하므로 로컬 개발은 Uvicorn으로 진행한다.

___

## 📌 코드 컨벤션

협업 시 코드 스타일을 통일하기 위해 다음 도구를 사용한다.

| 구분 | `pylint` | `black` |
| --- | --- | --- |
| 역할 | 코드 품질 검사 | 코드 자동 포매터 |
| 기능 | 문법 오류·미사용 변수·네이밍 탐지 | 들여쓰기·공백·따옴표 자동 정리 |
| 결과 | 문제 보고서 출력 | 코드 직접 수정 |
