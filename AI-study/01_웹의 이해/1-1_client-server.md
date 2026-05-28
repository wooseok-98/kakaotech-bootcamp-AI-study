# Client - Server

클라이언트가 요청(Request)을 보내면, 서버가 응답(Response)을 반환하는 구조.

- **Client**: 요청하는 주체 (브라우저, 앱 등)
- **Server**: 요청을 처리하고 결과를 반환하는 주체

---

## 네트워크 기본 개념

| 용어 | 설명 |
| --- | --- |
| IP Address | 서버의 위치를 나타내는 숫자 주소 (예: 192.168.0.1) |
| Domain | IP를 사람이 읽기 쉽게 변환한 문자 주소 (예: www.google.com) |
| Port | 서버와 통신하는 입구. 0 ~ 65535번까지 존재 |
| Firewall | 허용되지 않은 접근을 차단하는 보안 계층 |
| Protocol | 통신 규약. 웹에서는 주로 **HTTP/HTTPS** 사용 |

---

## 인증 vs 인가

| 구분 | 설명 |
| --- | --- |
| 인증 (Authentication) | 사용자가 누구인지 확인 (로그인) |
| 인가 (Authorization) | 인증된 사용자가 특정 리소스에 접근할 권한이 있는지 확인 |

---

## Client Side vs Server Side

**Client Side** — 브라우저에서 실행

| 언어 | 역할 |
| --- | --- |
| HTML | 페이지 구조 |
| CSS | 스타일·디자인 |
| JavaScript | 동적 동작 |

**Server Side** — 서버에서 실행

| 언어 | 특징 |
| --- | --- |
| Python | 범용성 높음, AI/ML 생태계 강점 |
| Java | 대규모 시스템에 강함 |
| JavaScript (Node.js) | 실시간 통신에 강함 |
| PHP | 간단한 웹에 사용 |

---

## Database

데이터를 저장·조회·관리하는 시스템.  
직접 접근하지 않고 **DBMS**(MySQL, PostgreSQL 등)를 통해 SQL 쿼리로 조작.

---
