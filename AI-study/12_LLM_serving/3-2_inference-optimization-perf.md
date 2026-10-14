# 3-2. 추론 최적화 ② 성능 지표·병렬화 (Throughput · Latency · Tensor Parallelism · Speculative Decoding)

LLM 서빙의 성능을 **숫자로 측정(Throughput·Latency)** 하고, GPU를 **나눠 쓰거나(TP) 생성을 앞당겨(Speculative Decoding)** 성능을 끌어올리는 기법

> 3-1이 "메모리를 어떻게 아끼나"였다면, 여기는 "그 결과 성능이 얼마나 나오나 + 어떻게 더 높이나"

---

## Throughput (처리량)

LLM 서빙 시스템이 **단위 시간 동안 여러 요청을 처리하며 생성한 출력 토큰의 총량** — 개별 요청 하나가 아니라 시스템 전체가 1초에 만든 결과의 총합

> 웹의 RPS(초당 처리 요청 수)에 해당하되, LLM은 요청마다 응답 길이가 달라 **토큰 단위**로도 봄. 값이 높을수록 같은 GPU로 더 많이 처리

### 측정 단위

| 단위 | 의미 | 사용 맥락 |
|---|---|---|
| `requests/s` | 초당 완료한 요청 수 | 요청 처리량 (웹의 RPS) |
| `output tokens/s` | 시스템 전체가 초당 생성한 출력 토큰 수 | **LLM 응답 생성량 (핵심 지표)** |
| `total tokens/s` | 입력+출력 합쳐 초당 처리한 전체 토큰 수 | 전체 토큰 처리량 |
| 요청당 `tokens/s` | 개별 요청의 초당 생성 토큰 수 | 개별 응답 속도 |

> 요청 수를 볼 땐 `requests/s`, **실제 생성량을 볼 땐 `output tokens/s`** — 같은 요청 수라도 응답 길이가 다르면 토큰 수는 다름

### 사용 이유 — 성능·비용·스케일링 판단

- **GPU 비용과 직결**: GPU는 사용 시간 기준 과금 → Throughput↑면 더 적은 GPU·짧은 시간으로 같은 트래픽 처리
- **설정 비교 기준**: `--max-num-seqs`·배칭·양자화·모델 크기·GPU 종류를 바꿔 어떤 설정이 더 많은 토큰을 내는지 비교
- **스케일링 계획**: 목표 `10,000 tokens/s` ÷ GPU 1장 `2,000 tokens/s` = 최소 5장 (+ 장애 여유·피크 고려)

> **Throughput만 높다고 좋은 건 아님** — 동시 요청·배치를 키우면 전체 처리량↑지만 개별 사용자 Latency↑. 실무는 **허용 지연시간 안에서 Throughput 최대화**

### 사용 방법 — 부하 테스트로 측정

서버를 포화시켜야 최대 Throughput이 드러남 (1건씩 보내면 GPU가 놀아 낮게 측정) → **동시 요청 수·입출력 길이를 함께 기록**해야 의미 있음

| 순서 | 단계 | 설명 |
|---|---|---|
| 1 | 부하 도구 준비 | vLLM 내장 벤치 또는 Locust·k6 |
| 2 | 동시 요청 전송 | 일정 시간 동안 N개 동시 요청 지속 |
| 3 | 토큰·시간 측정 | `usage.completion_tokens`로 토큰, 시작·종료 시각으로 시간 기록 |
| 4 | `tokens/s` 산출 | 전체 생성 토큰 ÷ 소요 시간 |

**① vLLM 내장 벤치 (`vllm bench serve`)**

```bash
vllm bench serve \
    --endpoint-type openai-chat --host localhost --port 8000 \
    --endpoint /v1/chat/completions --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset-name random --random-input-len 128 --random-output-len 128 \
    --num-prompts 200 --request-rate 10   # 초당 10개씩 총 200요청, 입출력 128토큰 고정
```

```
Request throughput (req/s):        6.26      # 초당 완료 요청 수
Output token throughput (tok/s):   448.18    # ★ 초당 생성 출력 토큰 (핵심)
Total Token throughput (tok/s):    1249.58   # 입력+출력 합산
```

> 입출력 길이를 고정(`--random-*-len`)해야 결과 비교가 쉬움. 함께 출력되는 TTFT·TPOT·ITL은 Latency 지표(아래)

**② Python으로 직접 측정**

```python
def send_one(prompt):
    res = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[{"role": "user", "content": prompt}], max_tokens=100)
    return res.usage.completion_tokens          # 이번 응답의 생성 토큰 수

prompts = ["어댑터즈는 스타트업코드에서 제공하는"] * 50   # 조건 고정 위해 동일 프롬프트

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:   # 동시성 10
    token_counts = list(ex.map(send_one, prompts))
elapsed = time.time() - start
print(f"Throughput: {sum(token_counts) / elapsed:.1f} tokens/s")    # 전체 토큰 ÷ 시간
```

> `max_workers`(동시 요청)를 늘리면 일정 구간까지 Throughput↑ — GPU 메모리·설정 한계에 닿으면 정체되거나 Latency↑

## Latency (지연시간)

요청을 보낸 시점부터 **첫 토큰이 도착하거나 전체 응답이 완료되기까지** 걸리는 시간 — 요청 하나가 사용자에게 얼마나 빨리 응답되는지

> Throughput이 "시스템 전체가 초당 얼마나 처리하나"라면, Latency는 **"요청 하나가 얼마나 빨리 오나"** — LLM은 토큰을 하나씩 생성하므로 첫 조각 도착과 전체 완료를 구분해서 봄

### 세부 지표

| 지표 | 의미 | 사용자 체감 |
|---|---|---|
| **TTFT** (Time To First Token) | 요청 후 첫 토큰 도착까지 | 빈 화면을 보며 기다리는 시간 |
| **TPOT** (Time Per Output Token) | 첫 토큰 이후 토큰 간 평균 생성 간격 | 글자가 이어서 차오르는 속도 |
| **Total Latency** | 요청 후 전체 응답 완료까지 총 시간 | 답변 전체가 끝날 때까지 |

```
요청 전송 ──TTFT──▶ 첫 토큰 ──TPOT──▶ 다음 토큰 ──TPOT──▶ ... ──▶ 마지막 토큰
└──────────────────── Total Latency ────────────────────┘
```

> TPOT는 도구·문서에 따라 **ITL (Inter-Token Latency)** 로도 부름 — 둘 다 첫 토큰 이후 토큰 간 평균 간격

### 사용 이유 — 체감 속도·품질 관리

- 같은 답을 내도 첫 글자가 늦거나 텍스트가 천천히 이어지면 "느리다"고 체감 → TTFT(시작)·TPOT(이어짐)로 나눠 확인
- **Throughput vs Latency 트레이드오프**: 동시 요청·배치를 키우면 처리량↑지만 개별 Latency↑ → **허용 Latency 안에서 Throughput 최대화**가 실무 목표
- `TTFT 500ms 이내`·`TPOT 50ms 이내`처럼 기준을 정해 모델 크기·입력 길이·동시 요청·배치·GPU를 조정

### 사용 방법 — Streaming으로 구간 측정 + SLA 비교

> **SLA (Service Level Agreement)**: 서비스가 보장하는 응답 속도·성공률 같은 품질 기준. 고정값이 아니라 서비스 목적·사용자 기대·비용을 고려해 운영자가 정함 (예: `P95 TTFT 1초`·`P95 TPOT 50ms`·`성공률 99%`)

Streaming으로 요청해야 첫 토큰·이후 토큰 시각을 나눠 잴 수 있음. 한 번 값은 흔들리므로 **여러 번 측정 후 백분위(P50/P95/P99)** 로 판단

> **백분위**: 측정값을 빠른 순 정렬한 기준점. `P95 TTFT 1초` = 요청 100개 중 95개가 1초 내 첫 토큰 수신. **평균이 괜찮아도 P95/P99가 높으면 일부 사용자는 느린 응답** → 꼬리 지연 확인용

| 순서 | 단계 |
|---|---|
| 1 | 기준(TTFT·TPOT·Total) 정하기 |
| 2 | Streaming 요청 |
| 3 | 요청 전송·첫 토큰·이후 토큰 시각 기록 |
| 4 | TTFT·TPOT·Total 계산 |
| 5 | 반복 측정해 P50/P95/P99 산출 |
| 6 | SLA와 비교 |
| 7 | 초과 시 입력 길이·출력 길이·동시 요청·배치·모델 크기·GPU 조정 |

```python
start = time.time()
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=100, stream=True,
    stream_options={"include_usage": True})   # 마지막 chunk에 토큰 사용량 포함

ttft, completion_tokens = None, 0
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        if ttft is None:
            ttft = time.time() - start          # 첫 내용 chunk 도착 = TTFT
    if chunk.usage is not None:
        completion_tokens = chunk.usage.completion_tokens or 0   # 실제 출력 토큰 수
end = time.time()

# 출력 토큰 N개면 간격은 N-1개 → 첫 토큰 이후 생성시간 ÷ (N-1)
generation_time = end - start - ttft
tpot = generation_time / (completion_tokens - 1)
```

> chunk 수 ≠ 토큰 수 → TPOT는 chunk 수가 아니라 **서버 집계 `completion_tokens`** 로 계산. TTFT·TPOT는 클라이언트 관찰값이라 모델 연산 + HTTP·SSE 전송 시간 포함

### TTFT vs TPOT — 영향 요인이 다름

TTFT는 입력을 한 번에 처리하는 **Prefill**, TPOT는 토큰을 하나씩 만드는 **Decode**의 영향을 주로 받음

| 요소 | TTFT | TPOT |
|---|---|---|
| 입력 길이↑ | Prefill 연산↑ → 크게 느려짐 | KV Cache 참조↑ → 약간 느려짐 |
| 동시 요청↑ | 대기 큐 발생 → 느려짐 | 배치↑ → 생성 간격↑ |
| 모델 크기↑ | 첫 계산량↑ → 느려짐 | 토큰당 연산↑ → 느려짐 |
| 양자화 | 가중치↓ → 빨라짐 | 가중치 읽기↓ → 빨라짐 |
| Tensor Parallelism | 연산 분담 → 빨라짐 | GPU 간 통신 비용 → 개선 폭 제한적 |

## Tensor Parallelism (TP)

하나의 LLM을 **여러 GPU에 가중치 행렬을 나누어 올리고**, 각 GPU가 연산의 일부를 동시에 처리하는 병렬화 방식

> 단순 분산 저장이 아님 — 여러 GPU가 하나의 요청을 함께 처리하며, 레이어마다 **결과를 합치고 공유**해 하나의 모델처럼 동작

### 동작 원리 — 분할 계산 + All-Reduce

- LLM의 Transformer 레이어 안 Attention·MLP 가중치 행렬을 GPU별로 조각내 배치 → 각 GPU는 **자기 조각만 계산** → 부분 결과 생성
- 부분 결과를 합쳐야 다음 레이어로 넘길 완성된 값이 됨 → **All-Reduce** 통신으로 모든 GPU가 같은 최종 결과 공유

> 웹 워커는 요청을 독립 처리하지만, TP의 GPU들은 **한 요청을 함께** 처리 → 레이어마다 중간 결과 교환 필요 → **GPU 간 통신 속도가 관건** (느리면 분할 이점보다 통신 비용↑)

### 사용 이유 — 단일 GPU 메모리 한계 돌파

- 대형 모델은 가중치만으로 GPU 초과 (예: **70B FP16 ≈ 140GB** → A100/H100 80GB 1장에 안 올라감) → 여러 GPU에 나눠 올림
- 양자화는 정밀도를 낮추지만, TP는 **FP16/BF16 정밀도 유지**하며 대형 모델 서빙 가능한 대안
- 여러 GPU의 메모리·연산 자원 결합 → 더 긴 Context·더 많은 동시 요청 + Prefill 연산 분담

> **GPU 수에 비례해 성능이 오르진 않음** — 레이어마다 All-Reduce 통신이 반복돼 GPU↑면 통신 비용도↑. 모델 크기·메모리·동시 요청·통신 비용을 함께 보고 **필요한 만큼만** 사용

### 사용 방법 — `--tensor-parallel-size N`

vLLM 서버 실행 시 플래그 하나 추가 — 가중치 분할·All-Reduce 통신을 vLLM이 내부 처리, 클라이언트 코드는 GPU 1장일 때와 동일

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2 \   # 모델을 GPU 2장에 분할
    --port 8000
```

| 순서 | 단계 | 설명 |
|---|---|---|
| 1 | GPU 수 확인 | 같은 노드에 NVLink로 연결된 GPU 수 확인 |
| 2 | TP 크기 결정 | **Attention 헤드 수의 약수 & 보유 GPU 수 이하** |
| 3 | 서버 기동 | `--tensor-parallel-size N` 추가 → vLLM이 자동 분할 |
| 4 | 요청 전송 | 클라이언트는 평소대로 OpenAI 호환 요청 |

> **TP 크기 제약**: ① 보유 GPU 수 이하 ② Attention 헤드 수를 나누어떨어지게. 예) 헤드 32개 → TP 2·4·8 가능, **TP 3은 불가**(균등 분할 안 됨)
>
> **NVLink**: NVIDIA GPU 간 고속 연결(PCIe보다 빠름). TP는 레이어마다 중간 결과를 주고받으므로 GPU 간 연결 속도가 중요 → NVLink 연결 GPU가 유리

### 부가 개념

**All-Reduce** — 모든 GPU가 부분 값을 합치고(Reduce), 합쳐진 결과를 다시 전부 공유(All). Transformer 레이어가 순차 연결돼 앞 레이어 결과가 다음 입력이 되므로, 레이어마다 결과를 맞춰야 함. GPU 통신이라 비용 있음 → 반복될수록 연결 속도 중요

**가중치 분할 방향** — 큰 가중치 행렬을 나누는 두 방식:

| 구분 | Column Parallel | Row Parallel |
|---|---|---|
| 나누는 방향 | 열 방향 | 행 방향 |
| 입력 | 각 GPU가 같은 입력 받아 자기 열 조각 계산 | 입력도 GPU별로 나뉘어 들어감 |
| 출력 | GPU별 출력 조각으로 나뉘어 생성 | 부분 결과를 합쳐야 최종 출력 |
| 통신 | 다음 연산이 분할 출력 그대로 받으면 통신↓ | 합산 통신(All-Reduce) 필요 |

> 두 방식을 조합하면 중간 결과를 자주 모으지 않고 분할 상태를 유지한 채 다음 연산으로 넘길 수 있어 통신 절약

**Pipeline Parallelism** — TP와 다른 병렬화. **자르는 위치가 다름**:

| 구분 | Tensor Parallelism | Pipeline Parallelism |
|---|---|---|
| 나누는 기준 | 하나의 레이어 **안** 가중치 행렬 | 여러 레이어를 **구간별로** |
| GPU 역할 | 모든 GPU가 같은 레이어 일부를 함께 계산 | GPU별로 다른 레이어 구간 담당 |
| 통신 위치 | 레이어 내부에서 자주 발생 | 레이어 묶음 경계에서 발생 |
| 비유 | 큰 계산 하나를 나눠 계산 | 긴 처리 과정을 단계별로 나눠 처리 (주문 검증→결제→배송) |

## Speculative Decoding

작고 빠른 **Draft 모델이 다음 토큰 여러 개를 미리 추측**하고, 크고 정확한 **Target 모델이 한 번에 검증**해 Decode 속도를 높이는 추론 가속 기법

> Decode는 토큰 하나마다 Target 순전파 1회 필요 (4토큰 = 4회) → Draft가 후보를 앞질러 제안하면 Target이 여러 개를 한 번에 확정

### 동작 원리 — 추측 후 검증

1. **Draft 모델**이 다음에 올 후보 토큰 K개를 미리 생성 (최종 답 아님, Target이 검증할 후보)
2. **Target 모델**이 후보들을 하나씩 재생성하지 않고 **한 번의 계산으로 검증**
3. 앞에서부터 수락, **처음 거부되는 지점부터 Target이 직접 다시 생성**

| 구분 | 토큰1 | 토큰2 | 토큰3 | 토큰4 |
|---|---|---|---|---|
| Draft 제안 | 제안 | 제안 | 제안 | 제안 |
| Target 검증 | 수락 | 수락 | **거부** | 검증 중단 |
| 최종 | 사용 | 사용 | Target 재생성 | Draft 제안 폐기 |

> 토큰3 거부 시 뒤의 토큰4도 폐기 — 앞 토큰이 달라지면 다음 토큰도 달라지므로. **품질은 Target과 동일** — Draft는 후보만 제안하고 Target 기준으로 수락/거부하므로 최종 출력은 Target만 쓴 것과 같은 분포

### 사용 이유 — Latency↓ (품질 유지)

- Continuous Batching·PagedAttention이 **Throughput** 최적화라면, Speculative Decoding은 **개별 요청 Decode 시간을 줄이는 Latency 최적화** — 토큰 간격↓, 전체 답변 완료↑
- **배치가 작거나 동시 요청이 적을 때 유리** (GPU 여유 → Draft 추가 계산 부담↓) / 이미 바쁜 GPU에선 Draft 실행 비용이 부담 → 항상 빠르진 않음
- Draft 모델을 함께 올려야 해 **GPU 메모리 추가 필요** → 켜면 좋은 옵션이 아니라 개선 효과 vs 추가 비용을 비교해 적용

### 사용 방법 — `--speculative-config`

Target만 지정하던 vLLM 실행에 Draft 모델·추측 토큰 수를 JSON으로 추가. 클라이언트 코드는 동일

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-config '{"method": "draft_model",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "num_speculative_tokens": 5}' \
    --port 8000
```

| 항목 | 의미 |
|---|---|
| `method` | 추측 방식 (별도 Draft 모델 사용 = `draft_model`) |
| `model` | 추측 담당 Draft 모델 (작고 빠른 모델) |
| `num_speculative_tokens` | 한 번에 미리 제안할 토큰 수 (K) |

> K를 키우면 한 번에 확정할 토큰↑, 대신 추측이 빗나갔을 때 버리는 양도↑

### Draft 모델 선택 기준

아무 작은 모델이나 안 됨 — Target과 잘 맞아야 함:

| 기준 | 설명 |
|---|---|
| 토크나이저 호환 | Target과 같은 방식으로 토큰화해야 검증 가능 |
| 충분한 생성 속도 | Target보다 충분히 빨라야 함 (느리면 오히려 병목) |
| 출력 분포 유사성 | Target과 비슷하게 예측해야 수락률↑ (다르면 잦은 거부→폐기) |

> 실무는 **같은 계열의 더 작은 모델** 사용 — 예) Target `Llama-3.1-70B` → Draft `Llama-3.1-8B`, Target `Qwen2.5-1.5B` → Draft `Qwen2.5-0.5B`

---

> 다음: LLM 배포 — Docker Containerization·MIG·Monitoring [[4-1_llm-deployment]]
