# HTTP 기초

---

## 목차

1. [HTTP란 무엇인가](#1-http란-무엇인가)
2. [HTTP 요청 구조](#2-http-요청-구조)
3. [HTTP 응답 구조](#3-http-응답-구조)
4. [HTTP 메서드](#4-http-메서드)
5. [HTTP 상태 코드](#5-http-상태-코드)
6. [REST API](#6-rest-api)
7. [우리 코드에서 HTTP가 어떻게 쓰이는가](#7-우리-코드에서-http가-어떻게-쓰이는가)

---

## 1. HTTP란 무엇인가

**HyperText Transfer Protocol** — 클라이언트(브라우저)와 서버가 데이터를 주고받는 규칙

- 브라우저가 서버에 요청(Request)을 보내면 서버가 응답(Response)을 돌려줌
- 텍스트, 이미지, JSON 등 모든 종류의 데이터를 주고받을 수 있음
- 요청 한 번, 응답 한 번이 한 쌍 → 이후 연결 끊김 (무상태, Stateless)

---

## 2. HTTP 요청 구조

브라우저가 서버로 보내는 메시지의 구조

```
POST /users HTTP/1.1              ← 요청 라인: 메서드 + URL + HTTP 버전
Host: localhost:8000              ← 헤더: 어느 서버에 보내는지
Content-Type: application/json    ← 헤더: body가 JSON 형식임을 알림

{                                 ← body: 실제 데이터
    "email": "test@test.com",
    "password": "1234",
    "nickname": "테스트"
}
```

### 요청의 구성 요소

| 구성 | 설명 | 예시 |
|------|------|------|
| 메서드 | 어떤 행동인지 | `GET`, `POST`, `DELETE` |
| URL | 어디에 요청하는지 | `/users`, `/posts/1/comments` |
| 헤더 | 요청에 대한 부가 정보 | `Content-Type: application/json` |
| Body | 전달하는 실제 데이터 | `{"email": "...", "password": "..."}` |

> GET 요청은 body가 없음. 데이터를 URL에 담음 → `/posts?page=1&limit=10`

---

## 3. HTTP 응답 구조

서버가 브라우저로 돌려주는 메시지의 구조

```
HTTP/1.1 200 OK                   ← 상태 라인: HTTP 버전 + 상태 코드 + 설명
Content-Type: application/json    ← 헤더

{                                 ← body: 실제 응답 데이터
    "message": "회원가입 성공",
    "user": {
        "id": 1,
        "email": "test@test.com",
        "nickname": "테스트"
    }
}
```

### 응답의 구성 요소

| 구성 | 설명 |
|------|------|
| 상태 코드 | 요청이 성공인지 실패인지, 왜 실패인지 |
| 헤더 | 응답에 대한 부가 정보 |
| Body | 실제 응답 데이터 (JSON, HTML 등) |

---

## 4. HTTP 메서드

메서드는 "이 요청이 어떤 행동인지"를 나타냄

| 메서드 | 행동 | 예시 |
|--------|------|------|
| `GET` | 데이터 조회 | 게시글 목록 가져오기 |
| `POST` | 데이터 생성 | 게시글 작성, 회원가입 |
| `PUT` | 데이터 전체 수정 | 게시글 전체 내용 교체 |
| `PATCH` | 데이터 일부 수정 | 게시글 제목만 수정 |
| `DELETE` | 데이터 삭제 | 게시글 삭제 |

### PUT vs PATCH

```
기존 데이터: { "title": "제목", "content": "내용", "views": 10 }

PUT  { "title": "새 제목" }
→ { "title": "새 제목" }  # content, views가 사라짐 (전체 교체)

PATCH  { "title": "새 제목" }
→ { "title": "새 제목", "content": "내용", "views": 10 }  # 나머지는 유지
```

---

## 5. HTTP 상태 코드

응답에 포함되어 "요청이 어떻게 됐는지"를 알려주는 숫자 코드

### 분류

| 범위 | 의미 |
|------|------|
| 2xx | 성공 |
| 3xx | 리다이렉트 (다른 URL로 이동) |
| 4xx | 클라이언트 오류 (요청이 잘못됨) |
| 5xx | 서버 오류 (서버 쪽 문제) |

### 자주 쓰는 상태 코드

| 코드 | 이름 | 사용 상황 |
|------|------|----------|
| 200 | OK | 일반적인 성공 |
| 201 | Created | 데이터 생성 성공 |
| 307 | Temporary Redirect | 다른 URL로 임시 이동 |
| 400 | Bad Request | 요청 데이터가 잘못됨 (빈 필드, 형식 오류) |
| 401 | Unauthorized | 로그인이 필요함 |
| 403 | Forbidden | 로그인은 됐지만 권한 없음 |
| 404 | Not Found | 존재하지 않는 리소스 |
| 409 | Conflict | 중복 데이터 (이메일 중복 등) |
| 500 | Internal Server Error | 서버 내부 오류 |

### 4xx vs 5xx 구분이 중요한 이유

- **4xx** → 클라이언트가 고쳐야 함 (입력값 확인, 로그인 등)
- **5xx** → 서버 개발자가 고쳐야 함

4xx인데 500을 반환하면 클라이언트는 자기 코드가 잘못됐는지 모름

---

## 6. REST API

**RE**presentational **S**tate **T**ransfer — HTTP를 활용해 API를 설계하는 규칙

### 핵심 원칙

**1. URL은 자원(명사)을 나타낸다**

```
# 나쁜 예 — 행동을 URL에 포함
POST /createUser
POST /deletePost?id=1
GET  /getUserList

# 좋은 예 — 자원만 URL에, 행동은 메서드로
POST   /users          # 유저 생성
DELETE /posts/1        # 게시글 삭제
GET    /users          # 유저 목록
```

**2. 계층 구조로 관계를 표현한다**

```
/posts              # 게시글 전체
/posts/1            # 1번 게시글
/posts/1/comments   # 1번 게시글의 댓글들
/posts/1/likes      # 1번 게시글의 좋아요
```

**3. 메서드로 행동을 구분한다**

```
GET    /posts        # 게시글 목록 조회
POST   /posts        # 게시글 생성
GET    /posts/1      # 1번 게시글 조회
DELETE /posts/1      # 1번 게시글 삭제
```

같은 URL에 메서드만 다르게 → 코드가 깔끔하고 의미가 명확해짐

---

## 7. 우리 코드에서 HTTP가 어떻게 쓰이는가

### 프론트엔드에서 요청 보내기

```javascript
// POST /posts/1/comments — 댓글 작성
const res = await fetch('/posts/1/comments', {
    method: 'POST',                              // 메서드
    headers: { 'Content-Type': 'application/json' },  // 헤더
    body: JSON.stringify({                       // body
        content: '댓글 내용',
        user_id: 1
    })
});
```

### FastAPI에서 요청 받기

```python
# POST /posts/{post_id}/comments 와 매핑
@router.post("/{post_id}/comments")
def create_comment(post_id: int, data: dict):
    # post_id → URL에서 자동으로 추출
    # data    → body의 JSON을 자동으로 파싱
    return comment_controller.create_comment(post_id, data)
```

### 상태 코드로 결과 전달

```python
# 성공 → FastAPI가 자동으로 200 반환
return {"message": "댓글 작성 성공", "data": new_comment}

# 실패 → 상태 코드 명시
raise HTTPException(status_code=404, detail="존재하지 않는 게시글입니다.")
raise HTTPException(status_code=409, detail="이미 좋아요를 눌렀습니다.")
```

### 우리 API 전체 목록 (REST 설계 기준)

| 메서드 | URL | 기능 |
|--------|-----|------|
| POST | `/users` | 회원가입 |
| POST | `/users/login` | 로그인 |
| GET | `/posts/` | 게시글 목록 |
| POST | `/posts/` | 게시글 작성 |
| GET | `/posts/{post_id}` | 게시글 조회 |
| DELETE | `/posts/{post_id}` | 게시글 삭제 |
| GET | `/posts/{post_id}/comments` | 댓글 목록 |
| POST | `/posts/{post_id}/comments` | 댓글 작성 |
| DELETE | `/posts/{post_id}/comments/{comment_id}` | 댓글 삭제 |
| GET | `/posts/{post_id}/likes` | 좋아요 수 조회 |
| POST | `/posts/{post_id}/likes` | 좋아요 추가 |
| DELETE | `/posts/{post_id}/likes` | 좋아요 취소 |