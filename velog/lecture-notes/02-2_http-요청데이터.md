<!-- 📝 글 소개 (출간 시 입력): HTTP 메시지·메서드·상태코드와 FastAPI 요청 데이터 처리 정리 -->

# 🎈 HTTP와 요청 데이터

**HTTP**는 웹 서버와 클라이언트가 데이터를 주고받기 위한 통신 규약이다. 요청(Request)과 응답(Response)이 모두 동일한 메시지 구조를 가진다.

___

## 📌 HTTP Message 구조

| 구성 요소 | 설명 |
| --- | --- |
| **Start Line** | 요청 메서드·URL·HTTP 버전 또는 응답 상태코드 |
| **Headers** | 메시지 부가 정보 (Content-Type, Host 등) |
| **Empty Line** | 헤더와 본문을 구분하는 빈 줄 |
| **Body** | 실제 전송 데이터 (JSON, HTML 등). GET 요청은 보통 없음 |

![HTTP 메시지 구조](여기에-이미지-드래그)

주요 헤더로는 본문 형식을 나타내는 `Content-Type`, 인증 토큰을 담는 `Authorization`, 요청 대상 서버를 가리키는 `Host` 등이 있다.

___

## 📌 HTTP Request Method

서버에 어떤 작업을 요청할지 나타내는 동사다. 같은 URL이라도 **메서드에 따라 동작이 달라진다**. 예를 들어 `/posts`에 `GET`을 보내면 조회, `POST`를 보내면 생성이다.

메서드는 두 가지 성질로 구분한다.

- **안전(Safe)**: 데이터를 변경하지 않음 → `GET`
- **멱등(Idempotent)**: 여러 번 호출해도 결과가 같음 → `GET`, `PUT`, `DELETE`

`POST`는 호출할 때마다 새 데이터가 생기므로 **멱등하지 않다**. 따라서 네트워크 오류로 재요청할 때 `GET`·`PUT`은 안전하지만 `POST`는 중복 생성에 주의해야 한다.

수정 메서드인 `PUT`과 `PATCH`도 구분된다. **`PUT`은 리소스 전체를 교체**하고, **`PATCH`는 일부 필드만 변경**한다.

| 메서드 | 용도 | 데이터 위치 |
| --- | --- | --- |
| `GET` | 데이터 조회 | Query String (URL) |
| `POST` | 데이터 생성 | Request Body |
| `PUT` | 데이터 전체 수정 | Request Body |
| `PATCH` | 데이터 부분 수정 | Request Body |
| `DELETE` | 데이터 삭제 | URL (리소스 식별자) |

```
GET    /posts        # 목록 조회
POST   /posts        # 생성
PATCH  /posts/1      # 부분 수정
DELETE /posts/1      # 삭제
```

___

## 📌 HTTP Status Code

서버가 요청을 처리한 결과를 나타내는 3자리 숫자 코드다.

| 번호대 | 의미 |
| --- | --- |
| `2xx` | 성공 |
| `3xx` | 리다이렉션 |
| `4xx` | 클라이언트 오류 |
| `5xx` | 서버 오류 |

| 코드 | 의미 | 예시 |
| --- | --- | --- |
| `200` | 성공 | 조회 성공 |
| `201` | 생성 성공 | 회원가입, 게시글 작성 |
| `204` | 성공 (반환 없음) | 수정, 삭제 성공 |
| `400` | 잘못된 요청 | 필수 파라미터 누락 |
| `401` | 인증 필요 | 로그인 안 한 접근 |
| `403` | 권한 없음 | 일반 유저의 관리자 접근 |
| `404` | 리소스 없음 | 삭제된 게시글 조회 |
| `500` | 서버 내부 오류 | 예상치 못한 에러 |

___

## 📌 URL 구조

```
https://www.example.com:443/posts/1?sort=latest
│       │               │   │       │
scheme  domain          port path   query string
```

| 구성 요소 | 설명 | 예시 |
| --- | --- | --- |
| Scheme | 사용할 프로토콜 | `https` |
| Domain | 요청 대상 서버 | `www.example.com` |
| Port | 접속 포트 (HTTP=80, HTTPS=443 생략 가능) | `443` |
| Path | 리소스 경로 | `/posts/1` |
| Query String | `?` 뒤 추가 파라미터, `&`로 구분 | `?sort=latest&page=2` |

___

## 📌 Query String vs Path Variable

URL로 데이터를 전달하는 두 가지 방식으로 나뉜다. 구분 기준은 **"자원을 식별하는가, 자원을 거르는가"**다. **Path Variable**은 특정 자원 하나를 가리킬 때(`/posts/123`), **Query String**은 목록을 조건으로 거를 때(`/posts?category=books`) 쓴다.

| 구분 | 형태 | 주요 용도 |
| --- | --- | --- |
| **Query String** | `/posts?category=books&page=1` | 검색, 필터링, 정렬, 페이지네이션 |
| **Path Variable** | `/posts/123` | 특정 리소스 식별 |

FastAPI에서는 함수 파라미터의 타입 힌트만으로 자동 검증된다. 기본값이 있으면 선택, 없으면 필수다.

```python
@app.get("/posts/{post_id}")           # Path Variable
def get_post(post_id: int):            # /posts/abc 요청 시 422 자동 반환
    ...

@app.get("/posts")                     # Query String
def get_posts(category: str, page: int = 1):
    ...
```

값 범위 검증이 필요하면 `Query`를 사용한다. 조건 불만족 시 `422`를 자동 반환한다.

```python
from fastapi import Query

@app.get("/posts")
def get_posts(page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=50)):
    ...
```

___

## 📌 JSON

**JSON**은 클라이언트와 서버가 데이터를 주고받는 표준 텍스트 포맷이다. 언어에 독립적이라 대부분의 프로그래밍 언어에서 지원한다.

```json
{
    "name": "wayne",
    "age": 25,
    "hobbies": ["reading", "hiking"]
}
```

FastAPI는 `dict`/`list`를 반환하면 자동으로 JSON으로 변환한다. 상태 코드를 직접 지정하려면 `JSONResponse`를 사용한다.

```python
from fastapi.responses import JSONResponse

@app.get("/items")
def get_items():
    return JSONResponse(content={"data": [...]}, status_code=201)
```
