# 02_FastAPI — 커뮤니티 서비스 백엔드

FastAPI로 커뮤니티 서비스 REST API를 구현하는 과제입니다.
동일한 기능을 세 가지 방식으로 단계별로 구현합니다.

---

## 버전 구성

| 폴더 | 데이터 저장 방식 | 특징 | 상태 |
|------|----------------|------|------|
| `code_vanilla/` | Python 리스트 (메모리) | Route-Controller-Model 패턴 학습 | ✅ 완료 |
| `code_DB/` | SQLite + SQLModel | 데이터 영구 저장, 작성자 표시 | ✅ 완료 |
| `code_AI/` | SQLite + SQLModel + Ollama | 게시글/댓글 AI 요약 기능 추가 | ✅ 완료 |

---

## 공통 기능

- 회원가입 / 로그인
- 게시글 CRUD
- 댓글 작성 / 삭제
- 좋아요 추가 / 취소
- 프론트엔드 (`index.html`) — 브라우저에서 직접 테스트 가능

---

## 실행 방법

각 버전의 폴더 안에서 실행합니다.

```bash
cd code_vanilla        # 또는 code_DB, code_AI
source .venv/bin/activate
uvicorn main:app --reload
```

- 프론트엔드: `http://localhost:8000`
- API 문서: `http://localhost:8000/docs`

> `code_AI` 실행 전 Ollama 설치 및 `ollama pull gemma3:1b` 필요

---

## 폴더 구조

```
02_FastAPI/
├── index.html                    # 공통 프론트엔드
├── README.md
│
├── code_vanilla/                  # v1: 메모리 기반
│   ├── main.py
│   ├── routers/
│   ├── controllers/
│   └── models/
│
├── code_DB/                      # v2: DB 연동
│   ├── main.py
│   ├── database.py
│   ├── routers/
│   ├── controllers/
│   └── models/
│
└── code_AI/                      # v3: AI 기능 추가
    ├── main.py
    ├── database.py
    ├── ai/
    │   └── llm.py
    ├── routers/
    ├── controllers/
    └── models/
```

---

## 설계 원칙

모든 버전에서 **Route-Controller-Model** 패턴을 사용합니다.

```
요청 → Router → Controller → Model → 응답
```

- `routers/` : URL과 함수 연결
- `controllers/` : 비즈니스 로직
- `models/` : 데이터 접근 (버전마다 구현 방식이 달라지는 부분)

버전이 올라갈수록 기존 코드는 최대한 유지하고 필요한 부분만 추가/교체합니다.

---

## 회고

<details>
<summary>code_vanilla — Route-Controller-Model 패턴 구현</summary>

### 잘 된 것
- Router → Controller → Model 3계층으로 역할을 분리해서, 각 파일이 하는 일이 명확해졌다.
- FastAPI의 `HTTPException`으로 상태 코드를 직접 지정하면서 4xx와 5xx의 차이를 실제로 체감했다.

### 어려웠던 것
- FastAPI를 처음 사용해봐서 처음 구조를 파악하는데 시간이 오래 걸렸다.
- 처음 구조를 공부하며 vanila 코드를 작성했고, 추후 한 단계씩 기능을 추가하며 구현했다.
- 코드를 작성하며 claude code의 도움을 받았지만, 코드를 작성해달라고 하기보다는 원리를 배우고, 스스로 구현하는 식으로 진행하였다.


</details>

<details>
<summary>code_DB — SQLModel로 데이터 영구 저장</summary>

### 잘 된 것
- `models/` 레이어만 교체했는데 `routers/`와 `controllers/` 코드가 그대로 동작했다. 계층 분리의 실제 효과를 체감했다.
- `session.commit()` → `session.refresh()` 패턴으로 DB가 자동 생성한 ID를 받아오는 흐름을 이해했다.

### 어려웠던 것
- `model_dump()`로 SQLModel 객체를 dict로 변환해야 컨트롤러 코드가 그대로 동작한다는 점을 처음에 몰랐다.
- ID를 수동으로 계산하던 코드(`len(...) + 1`)를 제거하고 DB 자동 증가로 바꾸는 과정에서 컨트롤러도 함께 수정해야 했다.

### 배운 것
- ORM을 쓰면 SQL을 몰라도 파이썬 문법으로 DB를 다룰 수 있다.
- 계층을 잘 나누면 저장 방식(메모리 → DB)이 바뀌어도 비즈니스 로직은 건드리지 않아도 된다.

</details>

<details>
<summary>code_AI — Ollama로 LLM 요약 기능 추가</summary>

### 잘 된 것
- `ai/llm.py` 한 파일에 Ollama 호출 로직을 모아두니, 나중에 모델을 바꾸거나 다른 LLM으로 교체할 때 이 파일만 수정하면 된다.
- 기존 코드를 하나도 건드리지 않고 새 라우터와 컨트롤러만 추가해서 기능을 붙였다.

### 어려웠던 것
- LLM 응답 속도가 느려서 처음에는 요청이 실패한 것처럼 보였다. `timeout=60`으로 설정해야 한다는 것을 알았다.
- 모델 크기가 작으면 한국어 요약 품질이 떨어지는 경우가 있었다.

### 배운 것
- FastAPI 서버가 클라이언트이기도 하다. 브라우저가 FastAPI에 요청하듯, FastAPI도 `httpx`로 Ollama에 요청을 보낸다.
- AI 서빙은 기존 백엔드 구조 위에 LLM 호출을 얹는 것이다. 완전히 새로운 개념이 아니라 HTTP 요청의 연장선이었다.

</details>

---

## 학습 노트

- [01_FastAPI_시작하기](./FastAPI_구현_정리_note/01_FastAPI_시작하기.md)
- [02_컨트롤러_확장하기](./FastAPI_구현_정리_note/02_컨트롤러_확장하기.md)
- [03_데이터베이스_적용하기](./FastAPI_구현_정리_note/03_데이터베이스_적용하기.md)
- [04_AI_모델_서빙](./FastAPI_구현_정리_note/04_AI_모델_서빙.md)
