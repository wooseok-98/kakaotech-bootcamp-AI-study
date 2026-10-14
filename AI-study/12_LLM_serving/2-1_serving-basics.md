# 2-1. 서빙 기초 구조 (OpenAI Compatible API · Streaming · Context Length · Max Tokens · Memory Usage · Structured Outputs)

LLM 서버를 실제로 호출하고 운영할 때 알아야 할 기본 인터페이스·응답 방식·토큰 한도

> vLLM 같은 서빙 엔진이 띄운 서버를 어떻게 호출하고, 응답을 어떻게 받고, 토큰 한도를 어떻게 설계하는지

---

## OpenAI Compatible API

OpenAI API와 **같은 요청·응답 형식**을 사용해, OpenAI SDK 코드로도 로컬·자체 서빙 모델을 호출할 수 있게 만든 API 형식

> **모델이 OpenAI라는 뜻이 아니라 호출 방식이 OpenAI API와 같다는 뜻** — 서버 뒤엔 vLLM·Ollama·llama.cpp가 띄운 로컬 모델이 있을 수 있음. `base_url`만 바꾸면 OpenAI SDK 그대로 사용

### Chat Completions 구조

| 구성 | 내용 |
|---|---|
| 엔드포인트 | `POST /v1/chat/completions` |
| 요청 필수 | `model`(모델명), `messages`(대화 메시지 배열) |
| 요청 선택 | `temperature`, `top_p`, `max_tokens`, `stream`, `stop` |
| 응답 | `id`, `model`, `choices`, `usage` |
| `choices` 내부 | `message.content`(답변), `finish_reason`(종료 이유) |

> 비유: **ORM** — MySQL→PostgreSQL 바꿔도 서비스 코드 유지하고 드라이버만 교체하듯, 클라이언트 호출 방식은 유지하고 **뒤의 서빙 엔진만 교체**

### 사용 이유

- 서빙 엔진 교체(llama.cpp 프로토타입 → vLLM 프로덕션 → TensorRT-LLM)가 잦은데, **엔진마다 API가 다르면 클라이언트 코드 전체 수정** → 공통 형식이면 `base_url`·모델명만 변경
- **생태계**: LangChain·LlamaIndex·Dify가 OpenAI 형식 기본 지원 → 적은 변경으로 연동

### 코드

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")  # 로컬 vLLM 서버

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=100, temperature=0.7,
)
print(response.choices[0].message.content)
```

> **완전 호환은 아님** — 기본 채팅은 대부분 동작하나 스트리밍·임베딩·함수 호출·구조화 출력 등은 엔진·버전마다 지원 범위가 다름 (여러 DB가 SQL을 지원해도 JSON 함수·확장 문법은 다른 것처럼). 문서 확인 필요

---

## Streaming Response

응답 전체를 완성한 뒤 한 번에 반환하지 않고, **토큰이 생성될 때마다 조각(chunk)을 순차 전달**하는 방식

### Non-streaming vs Streaming

| | Non-streaming | Streaming |
|---|---|---|
| 언제 보냄 | 응답 **전체 완성 후** 한 번에 | 토큰 생성될 때마다 조각씩 |
| 프로토콜 | 일반 REST(JSON) | **SSE** (Server-Sent Events) |
| 화면 | 완성까지 빈 화면 → 한 번에 | 생성되는 대로 이어붙여 표시 |

```
Non-streaming:  요청 → [5초 대기] → 전체 응답
Streaming:      요청 → 0.3초 → "어" → "댑" → "터" ... → [DONE]
```

> **총 생성 시간은 거의 동일** — 줄어드는 건 **체감 대기 시간**(빈 화면 5초 → 0.3초). ChatGPT처럼 한 글자씩 나타나는 UI. 도중 **중단(cancel)** 가능해 GPU 자원 절약

> SSE 진행 중 HTTP 연결은 열린 채 유지되지만, 세션·로그인 상태를 보관하는 **Stateful 서버는 아님** — 하나의 응답을 보내는 동안만 연결 유지

### 코드 (`stream=True`)

```python
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=100, stream=True,          # ← 핵심
)
for chunk in stream:
    content = chunk.choices[0].delta.content   # 이번 조각의 증분 텍스트 (전체 X)
    if content:
        print(content, end="", flush=True)     # 이어붙여 출력
```

> Non-streaming은 `message.content`(전체), Streaming은 `delta.content`(증분 조각). 종료 시 마지막 청크에 `finish_reason` + `data: [DONE]`

---

## Context Length

모델이 **한 번의 추론에서 처리할 수 있는 입력 토큰 + 출력 토큰의 최대 합** (모델 설계 시 정해지는 상한, 임의 확장 불가)

> **입력·출력이 한 한도를 나눠 씀** — 입력이 길수록 출력에 쓸 토큰↓. 예) Context 4,096에 입력 3,000이면 출력은 최대 1,096. Qwen2.5-1.5B은 32,768

> 비유: 한 API 요청의 **본문(payload) 크기 제한** — 넘으면 서버가 거부하거나 일부만 처리

### 초과하면?

| 상황 | 결과 |
|---|---|
| **입력**이 한도 초과 | **거부(400 에러)** 또는 조용히 **잘림**(보통 오래된 이력부터) — 잘림이 더 위험(품질만 조용히↓) |
| **출력**이 한도 도달 | 생성이 **문장 중간에서 강제 종료**, `finish_reason="length"` (정상 종료 `"stop"`과 구분) |

> **KV Cache는 요청별로 할당되고 실제 사용 토큰 수에 비례** → Context가 길수록 요청 하나가 수 GB 차지 가능 → **동시 요청 수 ↔ GPU 메모리** 균형 설계 필요. API는 입력·출력 토큰 모두 과금 → 비용 직결

### 사용 방법

`모델의 Context Length 확인 → 입력 예산 배분(시스템/이력/RAG) → 출력 여유 확보(max_tokens) → 초과 시 이력 요약·청크 축소·더 긴 모델`

```python
from transformers import AutoConfig, AutoTokenizer

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
print(config.max_position_embeddings)   # 32768 ← Context Length

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
n = len(tokenizer.encode("..."))         # 입력 토큰 수 (글자 수 아님)
```

> 토큰 수 ≠ 글자 수 — 토크나이저가 subword 단위로 쪼갬. **모델(토크나이저)마다 토큰 수가 달라** 반드시 해당 모델 토크나이저로 계산

### Context Length vs Max Tokens

| | Context Length | Max Tokens |
|---|---|---|
| 의미 | 입력+출력 **전체** 토큰 한도 | 새로 생성할 **출력** 토큰 한도 |
| 결정 | 모델 구조·설정 (고정) | 요청마다 사용자 지정 |
| 포함 | 시스템·사용자·이력·RAG·출력 전부 | 출력만 |
| 예시 | 4,096 / 32,768 / 128,000 | 256 / 512 / 1,024 |

> **그릇(Context) vs 출력 영역(Max Tokens)** — Context 4,096에 입력 3,000이면 `max_tokens=2,000`을 줘도 실제론 1,096만 (3,000+2,000 > 4,096). 둘 중 먼저 닿는 쪽에서 멈춤

> Context가 길다고 무조건 좋은 것 아님 — KV Cache·Prefill 시간↑, 동시 요청↓. **필요한 정보만 선별**해 넣는 게 실무 (RAG도 관련 청크만)

---

## Max Tokens

모델이 **한 요청에서 생성할 수 있는 출력 토큰의 최대 개수**를 지정하는 파라미터 (입력 토큰은 세지 않음, 출력만)

> 모델은 원래 EOS까지 생성하는데, `max_tokens`가 그 상한을 검 — 도달하면 EOS 전이라도 멈춤. 비유: 목록 API의 **`limit`** — 데이터가 많아도 limit만큼만 반환

### finish_reason으로 종료 이유 구분

| `finish_reason` | 의미 |
|---|---|
| `"stop"` | EOS 토큰 또는 지정 stop 문자열/토큰으로 **정상 종료** |
| `"length"` | `max_tokens` 상한에 걸려 **문장 중간에 잘림** |

> `max_tokens`는 상한이지 정확한 생성량이 아님 — `max_tokens=1000`이어도 50토큰에 EOS 나오면 50개만. 반대로 답변이 길면 중간에 잘리고 `"length"`

### 사용 이유

출력이 길수록 요청 하나가 **GPU 점유 시간↑·KV Cache↑·응답 시간↑·비용↑** → 한 요청이 전체 서빙 성능을 잡아먹지 않게 상한. API는 출력 토큰도 과금

### 사용 방법

용도별로 미리 설정 (사용자 질문마다 즉석 결정 X, **기능·엔드포인트 단위 기본값**):

| 작업 | max_tokens |
|---|---|
| 분류/판단 | 10~50 |
| 요약 | 100~500 |
| 자유 대화 | 500~2000 |

```python
r = client.chat.completions.create(model="...", messages=[...], max_tokens=10)
print(r.choices[0].finish_reason)   # "length" → 잘림 → 값 늘릴지 검토
```

> `finish_reason="length"`가 자주 나오면 상한을 늘림. 운영 데이터로 튜닝하는 값

---

## Memory Usage

LLM 서빙 시 **GPU VRAM**이 모델 가중치·KV Cache·활성화 값·프레임워크 오버헤드에 의해 점유되는 총량

> CPU RAM이 아니라 **GPU VRAM** 이야기. VRAM 초과 시 **OOM(Out of Memory)** → 서빙 중단

### 구성 요소

| 항목 | 성격 |
|---|---|
| **모델 가중치** | 모델 크기·dtype에 따라 **거의 고정** (7B FP16 ≈ 14GB) |
| **KV Cache** | 요청별, 시퀀스 길이·동시 요청 수에 따라 **가변** |
| 활성화 값 | 계산 중 임시 생성 |
| 오버헤드 | CUDA 컨텍스트·프레임워크 버퍼 (실제 모델보다 VRAM이 더 크게 보이는 이유) |

> 비유: 모델 가중치 = 서버 켤 때 항상 올라가는 **기본 메모리**, KV Cache = 요청마다 생겼다 사라지는 **요청별 임시 메모리**

### 알아야 하는 이유

- 모델 가중치만으로 VRAM 초과하면 **로드 단계에서 실패** (70B FP16 ≈ 140GB → A100 80GB 단일 불가 → 다중 GPU·양자화)
- Memory Usage 기준으로 **모델 크기·dtype·양자화·최대 문맥 길이·동시 요청 수**를 함께 조정 — 너무 크게 잡으면 OOM, 너무 보수적이면 처리량↓
- INT4 양자화하면 가중치 VRAM↓ → 남는 공간을 KV Cache로 → 더 긴 문맥·더 많은 동시 요청

### 추정 방법

`가중치 추정 → 요청별 KV Cache 추정 → 동시 요청 수 반영 → 오버헤드 추가 → GPU 용량과 비교`

```python
def weight_gb(param_b, bytes_per_param):     # 7B FP16 → 7 * 2 = 14GB
    return param_b * bytes_per_param

def kv_cache_gb(layers, heads, head_dim, seq_len, bytes_per_el):
    return 2 * layers * heads * head_dim * seq_len * bytes_per_el / 1e9   # K,V 두 텐서

w = weight_gb(7, 2)                                          # 14 GB
kv = kv_cache_gb(32, 32, 128, 2048, 2)                       # 1건 ≈ 1.07 GB
total = w + kv * 10 + 2                                       # 가중치 + KV(10건) + 오버헤드 ≈ 26.74 GB
```

> 실측 예: 7B FP16 = 가중치 14 + KV(10건, 2048토큰) 10.74 + 오버헤드 2 ≈ **26.74GB** → A100 40GB는 여유, RTX 3090 24GB는 **OOM** → 동시 요청↓·시퀀스↓·양자화

---

## Structured Outputs

응답을 **JSON Schema 등 사전 정의된 형식에 맞게 생성하도록 강제**해, 애플리케이션이 안정적으로 파싱할 수 있게 하는 기능

> 프롬프트에 "JSON으로 답해줘"는 **어길 수 있음**(중괄호 빠짐·필드명 다름·타입 오류). Structured Outputs는 **생성 단계에서** 형식 벗어난 토큰을 못 만들게 제한

### 프롬프트 요청 vs Structured Outputs

| | 프롬프트 요청 | Structured Outputs |
|---|---|---|
| 방식 | "JSON으로 답하세요" 지시 | **생성 시 허용 토큰 제한** |
| 형식 보장 | 모델이 어길 수 있음 | 정의된 형식만 생성 |
| 검증 시점 | 결과 나온 뒤 확인 | **생성 도중** 제한 |

> **Constrained/Guided Decoding**: Decode 단계에서 다음 토큰 후보 중 **형식 규칙(JSON Schema·정규식·선택지)에 맞는 토큰만** 남김. 예) JSON 시작 위치엔 `{`만, 필드명 위치엔 스키마 정의 필드명만

### 사용 이유

- LLM 응답은 끝이 아니라 **시작** — 파싱 → DB 저장 → 다른 agent·tool 호출로 이어지는 **파이프라인**을 뒷단 프로그램이 자동 처리
- 형식 깨지면 그 사슬이 무너짐(**후속 시스템 장애**) — vLLM은 200 OK인데 받은 코드가 터짐. 특히 agent 소비 시 사람 검수 없어 더 위험
- 형식 오류 → 재요청 → GPU 재사용 → **Throughput↓ Latency↑**. Structured Outputs로 재시도 감소

### 사용 방법 (vLLM)

| 방식 | 언제 | 예 |
|---|---|---|
| `response_format.json_schema` | 여러 필드 JSON 객체 | `{"company":..., "service":...}` |
| `extra_body.guided_regex` | 정해진 문자열 패턴 | `010-\d{4}-\d{4}` |
| `extra_body.guided_choice` | 정해진 선택지 중 하나 | `["긍정","부정","중립"]` |

```python
# ① JSON Schema — 객체 추출
r = client.chat.completions.create(
    model="...", messages=[...],
    response_format={"type": "json_schema", "json_schema": {
        "name": "entity_extraction",
        "schema": {"type": "object",
            "properties": {"company": {"type": "string"}, "service": {"type": "string"}},
            "required": ["company", "service"]}}},
)
result = json.loads(r.choices[0].message.content)   # 파싱 가능 보장

# ② 정규식 — 패턴 강제
client.chat.completions.create(model="...", messages=[...],
    extra_body={"guided_regex": r"010-\d{4}-\d{4}"})

# ③ 선택지 — 분류 강제
client.chat.completions.create(model="...", messages=[...],
    extra_body={"guided_choice": ["긍정", "부정", "중립"]})
```

> ⚠️ **보장하는 건 형식이지 값의 정확성이 아님** — `company`/`service` 필드가 든 유효 JSON은 보장되나, 작은 모델은 값을 잘못 채울 수 있음. 값 정확도는 모델 이해력·문맥에 달림
