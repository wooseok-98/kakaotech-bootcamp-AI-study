# code_AI — SQLite + AI 요약 기능 커뮤니티 API

`code_DB`에 Ollama LLM 연동 기능을 추가한 버전입니다.
게시글과 댓글을 AI로 요약하는 엔드포인트를 제공합니다.

---

## 사전 준비

Ollama가 설치되어 있어야 합니다.

```bash
# 모델 다운로드 (최초 1회)
ollama pull gemma3:1b
```

---

## 실행 방법

```bash
cd code_AI
source .venv/bin/activate
uvicorn main:app --reload
```

- API 문서: `http://localhost:8000/docs`
- 프론트엔드: `http://localhost:8000`

서버 최초 실행 시 `community.db` 파일이 자동으로 생성됩니다.

---

## 파일 구조

```
code_AI/
├── main.py               # FastAPI 앱 생성, 라우터 등록, 테이블 자동 생성
├── database.py           # SQLite 연결 설정 (community.db)
├── ai/
│   └── llm.py            # Ollama API 호출 (gemma3:1b)
├── routers/
│   ├── user_router.py
│   ├── post_router.py
│   ├── comment_router.py
│   ├── like_router.py
│   └── summary_router.py  # AI 요약 엔드포인트
├── controllers/
│   ├── user_controller.py
│   ├── post_controller.py
│   ├── comment_controller.py
│   ├── like_controller.py
│   └── summary_controller.py  # DB 조회 → LLM 호출 → 결과 반환
└── models/               # SQLModel 테이블 (code_DB와 동일)
    ├── user_model.py
    ├── post_model.py
    ├── comment_model.py
    └── like_model.py
```

---

## code_DB와 달라진 점

| 항목 | code_DB | code_AI |
|------|---------|---------|
| 추가된 파일 | — | `ai/llm.py`, `summary_controller.py`, `summary_router.py` |
| 추가된 패키지 | — | `httpx` |
| 그대로인 파일 | — | 나머지 전체 |

---

## 구현된 API

### 기존 API (code_DB와 동일)

| 기능 | 메서드 | URL |
|------|--------|-----|
| 회원가입 | POST | `/users` |
| 로그인 | POST | `/users/login` |
| 게시글 목록 | GET | `/posts/` |
| 게시글 작성 | POST | `/posts/` |
| 게시글 조회 | GET | `/posts/{post_id}` |
| 게시글 삭제 | DELETE | `/posts/{post_id}` |
| 댓글 목록 | GET | `/posts/{post_id}/comments` |
| 댓글 작성 | POST | `/posts/{post_id}/comments` |
| 댓글 삭제 | DELETE | `/posts/{post_id}/comments/{comment_id}` |
| 좋아요 조회 | GET | `/posts/{post_id}/likes` |
| 좋아요 추가 | POST | `/posts/{post_id}/likes` |
| 좋아요 취소 | DELETE | `/posts/{post_id}/likes` |

### 추가된 AI API

| 기능 | 메서드 | URL |
|------|--------|-----|
| 게시글 요약 | GET | `/posts/{post_id}/summary` |
| 댓글 요약 | GET | `/posts/{post_id}/comments/summary` |

---

## AI 요약 흐름

```
브라우저 → GET /posts/1/summary
               ↓
       summary_controller
               ↓
       DB에서 게시글 조회 (post_model)
               ↓
       Ollama에 요약 요청 (ai/llm.py)
               ↓
       요약 결과 반환
```

---

## 한계점

- 비밀번호가 평문으로 저장됨 (암호화 없음)
- 사용자 인증 토큰(JWT) 없음
- LLM 응답 속도가 느릴 수 있음 (모델 크기에 따라 다름)
