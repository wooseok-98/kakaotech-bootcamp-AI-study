# REST API

REST 아키텍처 스타일을 따르는 API  
HTTP 메서드 + 명사형 URL로 자원을 표현하고 조작하는 설계 방식

---

## REST 원칙

| 원칙 | 설명 |
| --- | --- |
| Uniform Interface | 일관된 URL 구조와 HTTP 메서드 사용 |
| Client-Server | 클라이언트와 서버를 독립적으로 분리 |
| Stateless | 서버는 클라이언트 상태를 저장하지 않음. 요청마다 필요한 정보를 모두 포함 |
| Cacheable | 응답은 캐싱 가능 여부를 명시 |

---

## URL 설계 규칙

| 규칙 | 좋은 예 | 나쁜 예 |
| --- | --- | --- |
| URI는 명사로 표현 | `/users` | `/getUsers` |
| 행위는 HTTP 메서드로 표현 | `POST /users` | `GET /createUser` |
| 계층 관계는 `/` 로 표현 | `/users/1/posts` | `/users-posts-1` |
| 소문자 사용 | `/users` | `/Users` |
| 하이픈(`-`) 사용, 밑줄(`_`) 사용 금지 | `/user-posts` | `/user_posts` |
| 확장자 사용 금지 | `/users/1/profile` | `/users/1/profile.png` |
| URI 끝에 슬래시(`/`) 금지 | `/users` | `/users/` |

---

## CRUD 예시

| 기능 | 올바른 설계 | 잘못된 설계 |
| --- | --- | --- |
| 목록 조회 | `GET /users` | `POST /getAllUsers` |
| 단건 조회 | `GET /users/1` | `GET /users/getById/1` |
| 생성 | `POST /users` | `GET /users/registerNewUser` |
| 전체 수정 | `PUT /users/1` | `GET /users/update/1` |
| 부분 수정 | `PATCH /users/1` | `POST /users/modify/1` |
| 삭제 | `DELETE /users/1` | `POST /users/delete/1` |