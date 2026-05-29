# code_vanila — 메모리 기반 커뮤니티 API

외부 DB 없이 Python 리스트(메모리)로 데이터를 저장하는 버전입니다.
FastAPI의 Route-Controller-Model 패턴을 익히는 데 집중한 구현입니다.

---

## 실행 방법

```bash
cd code_vanila
source .venv/bin/activate
uvicorn main:app --reload
```

- API 문서: `http://localhost:8000/docs`
- 프론트엔드: `http://localhost:8000`
    - 프론트엔드는 cluade code를 이용하여 구현

---

## 파일 구조

```
code_vanila/
├── main.py               # FastAPI 앱 생성, 라우터 등록, HTML 서빙
├── routers/              # URL 연결 (엔드포인트 정의)
│   ├── user_router.py
│   ├── post_router.py
│   ├── comment_router.py
│   └── like_router.py
├── controllers/          # 비즈니스 로직 (유효성 검사, 조건 처리)
│   ├── user_controller.py
│   ├── post_controller.py
│   ├── comment_controller.py
│   └── like_controller.py
└── models/               # 데이터 저장 및 조회 (메모리 리스트)
    ├── user_model.py
    ├── post_model.py
    ├── comment_model.py
    └── like_model.py
```

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

요청이 들어오면 다음 순서로 처리됩니다.

```
브라우저 → Router → Controller → Model → Controller → Router → 브라우저
```

- **Router**: 어떤 URL이 어떤 함수로 연결되는지만 정의
- **Controller**: 입력값 검사, 중복 확인, 데이터 조합 등 실제 로직
- **Model**: 데이터를 저장하거나 꺼내오는 역할 (현재는 리스트)

---

## 한계점

- 서버를 재시작하면 모든 데이터가 초기화됨
- 비밀번호가 평문으로 저장됨 (암호화 없음)
- 사용자 인증 토큰(JWT) 없음
