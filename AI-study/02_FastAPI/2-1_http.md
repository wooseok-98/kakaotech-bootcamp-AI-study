# HTTP

웹 서버와 클라이언트가 데이터를 주고받기 위한 통신 규약  
HTML, CSS, JS, 이미지 등 구조화된 텍스트를 전송하는 데 사용

---

# HTTP Message

클라이언트와 서버가 주고받는 통신의 기본 단위  
요청(Request)과 응답(Response) 모두 동일한 구조를 가짐

## 구조

| 구성 요소 | 설명 |
| --- | --- |
| Start Line | 요청 메서드·URL·HTTP 버전 또는 응답 상태코드 |
| Headers | 메시지에 대한 부가 정보 (Content-Type, Host 등) |
| Empty Line | 헤더와 본문을 구분하는 빈 줄 |
| Body | 실제 전송 데이터 (JSON, HTML 등). GET 요청은 보통 없음 |

## 주요 요청 헤더

| 헤더 | 설명 |
| --- | --- |
| `Host` | 요청 대상 서버 도메인 |
| `Content-Type` | 본문 데이터 형식 (예: `application/json`) |
| `Content-Length` | 본문 크기 (바이트) |
| `Authorization` | 인증 토큰 |
| `Accept` | 클라이언트가 처리 가능한 응답 형식 |

## 주요 응답 헤더

| 헤더 | 설명 |
| --- | --- |
| `Content-Type` | 응답 본문 데이터 형식 |
| `Server` | 서버 소프트웨어 정보 |

---

# HTTP Request Method

서버에 어떤 작업을 요청할지 나타내는 동사

| 메서드 | 용도 | 데이터 위치 |
| --- | --- | --- |
| `GET` | 데이터 조회 | Query String (URL) |
| `POST` | 데이터 생성 | Request Body |
| `PUT` | 데이터 전체 수정 | Request Body |
| `PATCH` | 데이터 부분 수정 | Request Body |
| `DELETE` | 데이터 삭제 | URL (리소스 식별자) |

### 사용 예시

```
GET    /posts          # 게시글 목록 조회
GET    /posts/1        # 게시글 단건 조회
POST   /posts          # 게시글 생성
PUT    /posts/1        # 게시글 전체 수정
PATCH  /posts/1        # 게시글 부분 수정 (제목만 변경 등)
DELETE /posts/1        # 게시글 삭제
```

---

# HTTP Status Code

서버가 요청을 처리한 결과를 나타내는 3자리 숫자 코드

| 번호대 | 의미 |
| --- | --- |
| `1xx` | 정보 (처리 중) |
| `2xx` | 성공 |
| `3xx` | 리다이렉션 |
| `4xx` | 클라이언트 오류 |
| `5xx` | 서버 오류 |

### 자주 쓰는 상태 코드

| 코드 | 의미 | 예시 |
| --- | --- | --- |
| `200` | 성공 | 조회 성공 |
| `201` | 생성 성공 | 회원가입, 게시글 작성 |
| `204` | 성공 (반환 데이터 없음) | 수정, 삭제 성공 |
| `400` | 잘못된 요청 | 필수 파라미터 누락 |
| `401` | 인증 필요 | 로그인 안 한 상태로 접근 |
| `403` | 권한 없음 | 일반 유저가 관리자 페이지 접근 |
| `404` | 리소스 없음 | 삭제된 게시글 조회 |
| `429` | 요청 횟수 초과 | 비밀번호 틀림 반복 |
| `500` | 서버 내부 오류 | 예상치 못한 서버 에러 |
| `503` | 서버 일시 사용 불가 | 트래픽 과부하, 점검 중 |

> 전체 목록: https://developer.mozilla.org/ko/docs/Web/HTTP/Status

---

# URL

웹 상의 리소스 위치를 나타내는 주소

```
https://www.example.com:443/posts/1?sort=latest
│       │               │   │       │
scheme  domain          port path   query string
```

| 구성 요소 | 설명 | 예시 |
| --- | --- | --- |
| Scheme | 사용할 프로토콜 | `https` |
| Domain | 요청 대상 서버 | `www.example.com` |
| Port | 서버 접속 포트. HTTP=80, HTTPS=443은 생략 가능 | `443` |
| Path | 리소스 경로 | `/posts/1` |
| Query String | 추가 파라미터. `?` 뒤에 작성, `&`로 구분 | `?sort=latest&page=2` |
