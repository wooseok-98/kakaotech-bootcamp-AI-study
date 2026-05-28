# 커뮤니티 서비스 백엔드 API

FastAPI를 사용해 커뮤니티 서비스의 백엔드 REST API를 구현한 과제입니다.

---

## 기능

| 기능 | 메서드 | URL |
|------|--------|-----|
| 회원가입 | POST | /users |
| 로그인 | POST | /users/login |
| 게시글 목록 | GET | /posts |
| 게시글 작성 | POST | /posts |
| 게시글 조회 | GET | /posts/{post_id} |
| 댓글 작성 | POST | /posts/{post_id}/comments |
| 좋아요 | POST | /posts/{post_id}/likes |

---

## 실행 방법

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

API 문서: `http://localhost:8000/docs`

---

## 파일 구조

```
02_FastAPI/
├── code/
│   ├── main.py
│   ├── routers/
│   │   ├── user_router.py
│   │   ├── post_router.py
│   │   ├── comment_router.py
│   │   └── like_router.py
│   ├── controllers/
│   │   ├── user_controller.py
│   │   ├── post_controller.py
│   │   ├── comment_controller.py
│   │   └── like_controller.py
│   └── models/
│       ├── user_model.py
│       ├── post_model.py
│       ├── comment_model.py
│       └── like_model.py
└── FastAPI_구현_정리_note/
    ├── 01_FastAPI_시작하기.md
    └── 02_컨트롤러_확장하기.md
```

---

## 설계 이유

### Route-Controller-Model 패턴 선택
역할별로 파일을 분리해 각 계층이 하나의 책임만 갖도록 설계했습니다.
- `routers/` — URL 연결만 담당
- `controllers/` — 비즈니스 로직 담당 (유효성 검사, 조건 처리)
- `models/` — 데이터 저장/조회 담당

기능이 추가될 때 관련 파일만 수정하면 되어 유지보수가 쉽습니다.

### DB 없이 메모리(리스트)로 구현
현재는 서버 재시작 시 데이터가 초기화되는 구조입니다.
추후 SQLite 또는 PostgreSQL 연동 시 `models/` 만 수정하면 됩니다.

---

## 앞으로 추가할 기능

- [ ] 게시글 수정 / 삭제
- [ ] 댓글 수정 / 삭제
- [ ] 데이터베이스 연동

---

## 회고

<details>
<summary>과제 회고</summary>

</details>
