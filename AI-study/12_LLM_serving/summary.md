# LLM Serving 한줄 정리

## 개요 (Inference · Serving · vLLM)

| 개념 | 한줄 정리 |
|------|------|
| LLM Inference | 학습 끝난 모델에 프롬프트를 넣어 토큰 단위로 텍스트를 생성하는 추론 과정 |
| Prefill | 입력 전체를 한 번에 처리해 KV cache 생성하는 단계(compute-bound) |
| Decode | KV cache를 참고해 토큰을 하나씩 순차 생성하는 단계(memory-bound) |
| Autoregressive Generation | 이전 토큰들을 보고 다음 토큰을 하나씩 예측해 이어붙이는 자기회귀 방식 |
| LLM Serving | 학습된 모델을 API 서버로 배포해 여러 요청마다 추론을 수행하는 시스템 |
| vLLM | Continuous Batching·PagedAttention으로 GPU 낭비를 줄여 더 많은 응답을 생성하는 오픈소스 서빙 엔진 |

## 서빙 기초

| 개념 | 한줄 정리 |
|------|------|
| OpenAI Compatible API | OpenAI와 같은 요청·응답을 사용해서 OpenAI SDK 코드에서 `base_url`만 바꿔 로컬 서빙 서버에 요청하는 호환 규격 |
| Streaming Response | 완성 응답을 한 번에 주지 않고 SSE로 토큰을 조각내 실시간 전송 |
| Context Length | 모델이 한번에 추론할 수 있는 입력+출력 토큰 합의 한도 |
| Max Tokens | 모델이 생성할 수 있는 출력 토큰 상한(입력 제외) |
| Memory Usage | GPU VRAM = 가중치(고정) + KV Cache(가변) + 오버헤드, 초과 시 OOM |
| Structured Outputs | 생성 단계에서 형식을 강제(json_schema·guided_regex·guided_choice) |

## 추론 최적화 ① 메모리·스케줄링

| 개념 | 한줄 정리 |
|------|------|
| KV Cache | 이전 토큰의 K/V를 GPU에 저장해 재사용 (연산 ↓, 메모리 ↑) |
| Continuous Batching | 매 Decode 스텝마다 끝난 요청을 빼고 대기 요청을 채워 GPU 슬롯을 비우지 않는 스케줄링 |
| PagedAttention | KV Cache를 고정 크기 블록으로 쪼개 비연속 메모리에 할당, OS 가상 메모리처럼 낭비↓ |
| Prefix Caching | 여러 요청이 공유하는 접두부 KV Cache를 요청 간 재사용해 Prefill 반복 계산↓, TTFT↓ |

> **GPU 배치 처리 원리** — GPU 한 장은 한 스텝에 **순전파 1번 = 배치 1개**만 계산, 동시성은 배치 **개수**가 아니라 **너비**(요청 여러 개를 하나의 큰 행렬로 묶음)에서 나옴
> - 요청 1개든 10개든 읽는 **가중치 행렬은 동일** → 놀던 코어를 채워 거의 같은 시간에 처리 (묶을수록 Throughput↑)
> - 무한정은 아님 — 요청마다 **KV Cache가 따로 쌓여**(메모리 천장), 포화점 넘으면 Latency만↑ → `--max-num-seqs`는 "채우라"가 아니라 **안전 상한**
> - **Continuous Batching**: 진행 중인 배치에 끝난 요청을 빼고 새 요청을 즉시 투입 (배치 전체 종료를 안 기다림) → 빈 슬롯을 안 놀림
> - 단 새 요청은 **Prefill(입력 전체 처리) 1회를 거친 뒤** 다음 스텝부터 Decode 대열 합류 → 그 무거운 Prefill이 기존 요청을 잠깐 지연 (**Prefill-Decode interference**)
> - 배치를 여러 개 돌리려면 GPU를 늘리거나 [[4-1_llm-deployment]]의 **MIG로 분할**

## 추론 최적화 ② 성능 지표·병렬화

| 개념 | 한줄 정리 |
|------|------|
| Throughput | 시스템 전체가 단위 시간에 생성한 출력 토큰 총량(output tokens/s), GPU 비용 효율 지표 |
| Latency | 요청 하나의 응답 속도, TTFT(첫 토큰)·TPOT(토큰 간격)·Total로 구분 |
| TTFT / TPOT | TTFT=Prefill 영향(첫 글자 대기) / TPOT=Decode 영향(글자 이어지는 속도) |
| Tensor Parallelism | 가중치 행렬을 여러 GPU에 나눠 올리고 All-Reduce로 합치는 병렬화, 단일 GPU 메모리 한계 돌파 |
| Speculative Decoding | 작은 Draft가 토큰 여러 개를 미리 추측하고 큰 Target이 일괄 검증, 품질 유지하며 Latency↓ |

## LLM 배포

| 개념 | 한줄 정리 |
|------|------|
| Docker Containerization | 서빙 앱과 실행 환경(CUDA·vLLM·라이브러리)을 이미지로 고정해 서버마다 동일 재현 |
| Multi-Instance GPU (MIG) | GPU 하나를 자체 SM·VRAM을 가진 여러 독립 인스턴스로 하드웨어 분할 |
| Monitoring | 운영 서버의 처리량·지연·GPU 메모리 지표를 `/metrics`→Prometheus 도구로 지속 관측해 장애 조기 대응 |
