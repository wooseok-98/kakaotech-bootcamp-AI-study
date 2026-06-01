# code_DB — SQLite 기반 커뮤니티 API

SQLModel + SQLite로 데이터를 영구 저장하는 버전입니다.
`code_vanila`에서 `models/` 레이어만 교체하여 DB를 적용했습니다.

---

## 실행 방법

```bash
cd code_DB
source .venv/bin/activate
uvicorn main:app --reload
```

- API 문서: `http://localhost:8000/docs`
- 프론트엔드: `http://localhost:8000`

서버 최초 실행 시 `community.db` 파일이 자동으로 생성됩니다.

---

## 파일 구조

```
code_DB/
├── main.py               # FastAPI 앱 생성, 라우터 등록, 테이블 자동 생성
├── database.py           # SQLite 연결 설정 (community.db)
├── routers/              # URL 연결 (code_vanila와 동일)
│   ├── user_router.py
│   ├── post_router.py
│   ├── comment_router.py
│   └── like_router.py
├── controllers/          # 비즈니스 로직 (code_vanila와 거의 동일)
│   ├── user_controller.py
│   ├── post_controller.py
│   ├── comment_controller.py
│   └── like_controller.py
└── models/               # 데이터 저장 및 조회 (SQLModel 테이블)
    ├── user_model.py
    ├── post_model.py
    ├── comment_model.py
    └── like_model.py
```

---

## code_vanila와 달라진 점

| 항목 | code_vanila | code_DB |
|------|-------------|---------|
| 데이터 저장 | Python 리스트 (메모리) | SQLite 파일 (`community.db`) |
| 서버 재시작 시 | 데이터 초기화 | 데이터 유지 |
| ID 생성 | 수동 (`len(...) + 1`) | DB 자동 증가 |
| 변경된 파일 | — | `database.py` 추가, `models/` 전체 교체 |
| 그대로인 파일 | — | `routers/` 4개, `comment_controller.py`, `like_controller.py` |

---

## 구현된 API

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

---

## 패턴 설명

```
브라우저 → Router → Controller → Model(SQLite) → Controller → Router → 브라우저
```

- **Router**: 어떤 URL이 어떤 함수로 연결되는지만 정의
- **Controller**: 입력값 검사, 중복 확인, 데이터 조합 등 실제 로직
- **Model**: SQLModel로 테이블 정의, Session으로 DB 읽기/쓰기

---

## 한계점

- 비밀번호가 평문으로 저장됨 (암호화 없음)
- 사용자 인증 토큰(JWT) 없음
