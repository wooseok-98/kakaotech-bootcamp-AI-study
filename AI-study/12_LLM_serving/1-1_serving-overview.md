# 1-1. LLM Serving 개요 (LLM Inference · LLM Serving · vLLM)

학습이 끝난 LLM을 **실제 사용자 요청에 응답하는 서비스**로 만드는 배포·운영 단계 — 10장 최적화가 "모델을 가볍게"였다면, 여기는 "가벼워진 모델을 **실제로 서빙**"

> 10장 노트가 계속 "LLM Deployment에서 다룸"이라 예고했던 그 내용 — KV Cache·Throughput/Latency·TP는 최적화 관점에서 이미 맛봤고, 여기선 **서빙 관점**으로 다시 봄

---

## LLM Inference

학습이 완료된 LLM에 프롬프트를 입력해 **토큰 단위로 텍스트를 생성**하는 과정 (= LLM 추론)

> 가중치를 고정한 채 순전파(forward pass)를 수행 — Training(가중치 수정)과 반대. 비유: Training=서비스 개발, Inference=배포된 서버가 요청에 응답

### Prefill과 Decode

추론은 내부적으로 두 단계

| 단계 | 하는 일 | 성격 |
|---|---|---|
| **Prefill** | 입력 프롬프트 **전체를 한 번에** 처리해 KV Cache 생성 + 첫 토큰 | 병렬 처리 → **compute-bound** |
| **Decode** | KV Cache 참조하며 토큰을 **하나씩 순차** 생성 | 매 토큰마다 가중치·KV 읽기 → **memory-bound** |

> **compute-bound**=연산 속도가 병목 / **memory-bound**=메모리 읽기 속도가 병목. Decode는 병렬화가 어려워 **서빙 지연시간·비용에 큰 영향**

### Autoregressive Generation (자기회귀 생성)

이전에 입력·생성된 토큰을 바탕으로 **다음 토큰을 하나씩** 예측 → 문장에 이어붙이며 반복 (EOS 토큰 또는 max_tokens까지)

> 본질적으로 **순차적** — 응답이 길수록 Decode 반복↑ → 시간↑. 일반 이미지 분류(순전파 1번)와 달리 **출력 토큰 수만큼 순전파 반복**

### 추론 흐름 & 샘플링

`모델 로드 → 토큰화 → Prefill → Decode → 종료 판정(EOS/max_tokens) → 디토큰화`

한 토큰을 고르는 내부 순서 (`generate()`가 자동 수행):
```
점수 매김(logit) → Temperature(점수 조절) → softmax(확률 변환) → top_p(후보 컷) → 추첨(최종 1개)
```

| 파라미터 | 역할 |
|---|---|
| **Temperature** | logit을 T로 나눠 분포 조절 — **낮을수록 뾰족(결정적)**, 높을수록 평평(다양) |
| **top_p** | 확률 높은 순 누적이 p에 닿을 때까지만 후보로, 나머지 버림 |
| **do_sample=False** | temperature·top_p·추첨 다 건너뛰고 **매번 1등(greedy)** |

> softmax는 켜고 끄는 게 아니라 **항상 도는 고정 변환** — logit을 합이 1인 확률로 바꿈. top_p 후 남은 후보를 재정규화(합=1) → 가중 추첨이라 같은 질문에도 답이 조금씩 달라짐

```python
outputs = model.generate(
    **inputs, max_new_tokens=50,
    temperature=0.7, top_p=0.9, do_sample=True,   # 샘플링 사용
)
```

---

## LLM Serving

학습된 LLM을 **API 서버 형태로 배포**해, 여러 클라이언트 요청마다 추론을 수행하고 결과를 반환하는 **시스템·과정**

> **Inference ⊂ Serving** — Inference는 연산 자체, Serving은 그걸 감싸 여러 사용자가 안정적으로 쓰게 만드는 서비스. 비유: **요리(Inference) vs 식당 운영 전체(Serving)**

### Inference vs Serving

| | Inference | Serving |
|---|---|---|
| 범위 | 입력→출력 연산 | 추론을 API로 제공하는 전체 시스템 |
| 관심사 | 토큰 생성 속도·메모리·KV Cache | 동시 요청·처리량·안정성·확장성·장애 대응 |
| 웹 비유 | 비즈니스 로직 함수 | 웹 서버 + 운영 인프라 |

### 서빙 시스템 구성 요소

| 구성 요소 | 역할 |
|---|---|
| **서빙 엔진** | 모델을 GPU에 올려 실행 + KV Cache·배칭·토큰 생성 최적화 (vLLM, TensorRT-LLM, llama.cpp) |
| **요청 스케줄러** | 여러 요청을 묶어 GPU 효율↑ (→ Continuous Batching) |
| **메모리 관리자** | VRAM에 가중치·KV Cache·중간 데이터 효율 배치 |
| API 서버 | HTTP 엔드포인트 제공 (`/v1/chat/completions`) |
| 로드밸런서 / 모니터링 / Health Check | 일반 백엔드와 동일 |

> API 서버·로드밸런서·모니터링은 일반 백엔드에도 있는 것 — **LLM 서빙의 새 부분은 서빙 엔진·요청 스케줄러·메모리 관리자** (토큰 생성·GPU 메모리·동시 처리와 직결)

### 사용 이유

- `model.generate()` 로컬 실행 ≠ **여러 사용자 동시 접속 서비스** — 동시 요청·대기열·타임아웃·장애 대응·모니터링 필요
- 핵심 문제 = **동시 요청 처리**: 하나씩 처리하면 앞 요청 끝날 때까지 대기 → 서빙 엔진이 묶어 처리
- **비용**: GPU는 사용 시간이 곧 비용 → 같은 GPU로 더 많은 요청 처리하면 요청당 비용↓

---

## vLLM

여러 요청을 동시에 처리하고 **GPU 메모리 낭비를 줄여**, 같은 GPU에서 더 많은 응답을 생성하는 **오픈소스 LLM 서빙 엔진**

> **v = virtual** (OS의 **가상 메모리** 페이징에서 따옴). 학습 도구가 아니라 **학습 끝난 모델을 서비스로 실행**하는 엔진

### 왜 빠른가 — 두 핵심

| 기술 | 원리 |
|---|---|
| **Continuous Batching** | 토큰 생성 스텝마다 여러 요청을 한 배치에 끼워넣음 — A가 300토큰째 만드는 동안 B의 첫 토큰도 같은 배치에서 처리 (요청 단위 GPU 점유 X) |
| **PagedAttention** | KV Cache를 OS 가상 메모리처럼 **작은 블록(페이지)으로 쪼개 실제 쓰는 만큼만 할당** — 최대 길이 통째 예약 안 함 → 남는 메모리로 더 많은 요청 |

> **KV Cache 관점 차이**: 10장(Optimization)은 "재계산 방지로 추론 가속", 여기(Serving)는 **"동시 요청 시 GPU 메모리를 얼마나 차지하나"** — 요청마다 KV Cache가 쌓여 동시 요청↑ = 메모리↑. PagedAttention이 이를 관리

### vs FastAPI 배포

> **계층이 다름** — FastAPI 배포 = 웹 프레임워크(FastAPI) + 추론 라이브러리(transformers) 조합 / vLLM = **추론 전용 엔진**. 단순 FastAPI+transformers는 개별 HTTP 요청을 자동 동적 배칭 안 함 → vLLM이 지속적으로 묶어 처리해 처리량↑

### 사용 방법 (코드)

`pip 설치 → CLI 한 줄로 OpenAI 호환 API 서버 실행`

```bash
pip install vllm openai
```

```bash
# CLI 한 줄로 서버 실행
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --dtype float16 \                  # T4는 bfloat16 미지원
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

```python
# 기존 OpenAI SDK 코드에서 base_url만 교체
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=100, temperature=0.7,
)
```

| 주요 CLI 옵션 | 역할 | 기본값 |
|---|---|---|
| `--port` | API 서버 포트 | 8000 |
| `--gpu-memory-utilization` | GPU 메모리 사용 비율 (너무 높으면 OOM) | 0.9 |
| `--max-model-len` | 최대 문맥 길이(입력+출력), 클수록 KV Cache↑ | 모델 설정값 |
| `--tensor-parallel-size` | 모델을 나눠 올릴 GPU 개수 | 1 |
| `--quantization` | awq·gptq·fp8 등 | None |
| `--dtype` | float16·bfloat16·auto (T4는 float16) | auto |

> **vLLM vs Ollama**: Ollama는 **로컬 개발·테스트**용(명령어 한 줄, 간단), vLLM은 **API 서버 운영**용(동시 요청·GPU 메모리 관리). 개발은 Ollama, 배포는 vLLM으로 전환하는 흐름 — 둘 다 OpenAI 호환 API 지원

> Prefill/Decode·KV Cache·Continuous Batching·PagedAttention·TP는 뒤 페이지에서 상세히 — [[3-1_inference-optimization-memory]], [[3-2_inference-optimization-perf]]
