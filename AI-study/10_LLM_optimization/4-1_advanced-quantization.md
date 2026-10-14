# 4-1. 고급 Quantization (GPTQ · AWQ · Quantization Evaluation)

단순 반올림(RTN)보다 **양자화 오차를 똑똑하게 줄이는** 고급 PTQ 알고리즘과, 양자화 결과를 검증하는 평가 방법

> 앞서 본 Outlier Problem·오차 누적을 실제로 해결하는 기법들 — 모두 **PTQ의 하위 알고리즘** (재학습 없음)

---

## GPTQ (Generative Pre-trained Transformer Quantization)

학습된 LLM 가중치를 INT4 같은 낮은 비트로 줄이되, **단순 반올림보다 출력 품질이 덜 떨어지도록 오차를 보정**하는 PTQ 기법

> PTQ가 큰 범주, GPTQ는 그 안의 구체적 알고리즘 — 재학습 없이 소량의 calibration 데이터만 사용

### RTN의 한계 (비교 기준)

**RTN (Round-to-Nearest)**: 각 가중치를 **독립적으로** 가장 가까운 정수로 반올림

> 구현 단순·빠르지만 어떤 가중치가 중요한지, 오차가 어떻게 연결되는지 무시 → INT8은 괜찮으나 **INT4처럼 칸이 적으면 오차 누적으로 품질 급락**

### 핵심 아이디어 — 오차를 버리지 않고 분배

> **"한 가중치를 양자화하며 생긴 오차를, 아직 양자화하지 않은 다른 가중치에 분배한다"**

레이어 전체 출력이 원본 FP16과 최대한 비슷하게 유지되도록, 앞에서 생긴 오차를 뒤쪽 가중치가 흡수

### 알고리즘 흐름

| 순서 | 단계 |
|---|---|
| 1 | Calibration 데이터(소량 대표 텍스트) 입력 |
| 2 | 레이어 입력 통계 수집 |
| 3 | **민감도 정보(Hessian) 계산** — 어떤 가중치 오차가 출력에 큰 영향? |
| 4 | 일부 가중치를 INT4로 양자화 |
| 5 | **오차 보상** — 생긴 오차를 미양자화 가중치에 반영 |
| 6 | 다음 가중치/block에 반복 |
| 7 | 전체 변환 완료 (INT4 가중치 + scale 저장) |

### Hessian — 민감도 지도

가중치가 바뀌었을 때 **출력이 얼마나 민감하게 흔들리는지** 알려주는 정보 (레이어 재구성 오차의 2차 정보)

| 위치 | GPTQ의 처리 |
|---|---|
| 출력에 **민감**한 가중치 | 양자화 오차를 최대한 줄임 |
| 출력에 **둔감**한 가중치 | 다른 곳의 오차를 일부 흡수 |

> 비유: **서비스 영향도 맵** — 결제·로그인처럼 조금만 바뀌어도 장애 나는 핵심 로직 vs 관리자 화면처럼 영향 작은 로직. 오차를 덜 위험한 위치로 분배

### damped inverse Hessian

Hessian 역행렬은 그대로 계산하면 **수치적으로 불안정** (작은 값·비슷한 값 섞이면 발산) → 작은 보정값을 더해 안정화(**dampening**) 후 역행렬 사용

> 민감도 지도를 계산하기 전에 **너무 흔들리지 않게 smoothing** 한 버전

### 사용 이유

- INT4로 메모리 크게 줄이면서 **출력 품질 유지** → 더 작은 GPU·로컬 추론 가능
- **생태계**: Hugging Face에 Llama·Mistral·Mixtral·Qwen·Yi 등의 GPTQ 버전 공개 → 직접 양자화 없이 다운로드해 바로 사용

### 사용 방법

`원본 모델 → calibration 데이터 → 옵션 설정 → 양자화 실행 → 저장 → 평가 → 배포`

| 핵심 설정 | 의미 |
|---|---|
| **비트 수** | 주로 `4bit` — 낮을수록 메모리↓ 정확도 손실 위험↑ |
| **Calibration 데이터** | 오차 민감도 판단 기준. 범용은 C4·WikiText-2, 도메인 특화는 해당 도메인 텍스트 |
| **그룹 크기** | `group_size=128` = 가중치 128개마다 별도 scale·zero-point. 작을수록 정확↑ 메타데이터↑ |
| 오차 보정 설정 | inverse Hessian 안정화, 처리 순서, 대칭/비대칭 등 |

> 실무: `group_size=128`을 출발점으로, 정확도 부족하면 `64`·`32` 검토. **calibration 데이터와 group_size가 가장 중요**

### 코드 (직접 양자화)

```python
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset

model_id, quant_path = "Qwen/Qwen2.5-0.5B-Instruct", "qwen2-5-0-5b-gptq-int4"

raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
calibration_dataset = [t for t in raw["text"] if t.strip()][:256]   # 대표 텍스트 256개

quant_config = QuantizeConfig(bits=4, group_size=128)
model = GPTQModel.load(model_id, quant_config)
model.quantize(calibration_dataset, batch_size=2)   # 통계 수집 → 양자화 → 오차 보상 → 패킹
model.save(quant_path)
# Pre-Quantized 0.92GB → Quantized 0.43GB (약 53% 감소)
```

```python
# 저장본 로드 & 추론
from gptqmodel import GPTQModel, BACKEND
model = GPTQModel.load(quant_path, backend=BACKEND.TORCH, device="cuda:0")
print(model.tokenizer.decode(model.generate("어댑터즈는")[0], skip_special_tokens=True))
```

```python
# 이미 양자화된 공개 모델 로드 (양자화 계산 없이 바로 추론)
model = GPTQModel.load("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", backend=BACKEND.TORCH, device="cuda:0")
```

> **BPW 4.2875**: 순수 4bit 가중치뿐 아니라 scale·zero-point 메타데이터까지 포함한 **실질 bits-per-weight**. 직접 양자화는 INT4 패킹이 느려 0.5B 모델도 T4에서 30~50분 소요 — 흐름만 볼 땐 사전 양자화 모델 사용 권장

> 논문: *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (Frantar et al., 2022.10, ICLR 2023)

---

## AWQ (Activation-aware Weight Quantization)

**활성화 통계**를 보고 출력에 중요한 가중치 채널을 찾아, 그 채널이 INT4 양자화에서 **덜 손상되도록 스케일을 조정**하는 PTQ 기법

> 가중치만 보지 않고 **활성화 분포를 함께 봄** — 어떤 가중치를 더 조심스럽게 양자화할지 판단

### 핵심 관찰

> **활성화 값이 큰 채널과 연결된 가중치일수록 출력에 더 큰 영향** — 같은 크기의 양자화 오차라도 활성화가 큰 채널에서 생기면 최종 출력에 더 크게 반영됨

| 관찰 | 의미 |
|---|---|
| 특정 입력 채널의 활성화 값이 큼 | 그 채널이 출력에 크게 기여 |
| 그 채널의 가중치를 거칠게 양자화 | 작은 가중치 오차도 출력에 크게 드러남 |
| 활성화 큰 채널의 가중치를 보호 | **적은 수의 가중치만 보호해도** 전체 오차 크게 감소 |

### Salient Weight

양자화 시 **출력에 큰 영향을 주는 중요 가중치**

> **핵심**: "값이 큰 가중치"가 아니라 **"연결된 채널의 활성화가 큰 가중치"** — 가중치 크기만 보고 상위 1%를 고르는 게 아니라, calibration으로 얻은 activation statistics 기준으로 판단

### 알고리즘 — Per-channel Scaling

중요 채널을 **FP16으로 따로 저장하지 않음** (정밀도 경로가 섞여 커널 복잡·메모리 오버헤드)

> 대신 중요 채널의 가중치에 **스케일을 곱해 값을 키운 뒤 양자화** → 같은 INT4 표현 안에서 더 의미 있는 값으로 매핑 (상대 오차↓)

```
① 중요 채널 가중치 × scale  → INT4 안에서 잘 표현됨
② 선형 연산 체인에서 반대 방향 scale 흡수 → 등가 변환 유지
   (양자화가 없다면 원래 출력과 동일한 결과)
```

> 스케일링은 출력을 바꾸는 조작이 아니라, **양자화 전에 중요 가중치가 덜 뭉개지도록 만드는 준비 단계**

### 사용 이유

- 별도 **학습 없이** 적용, 역전파·레이어 재구성에 의존 안 함
- Hessian 계산 대신 **활성화 통계만** → 워크플로 단순, 계산 부담 가벼움
- 고정밀 경로를 늘리지 않아 **추가 메모리 오버헤드도 작음**

### 사용 방법

`원본 모델 → calibration → 활성화 통계 수집 → 중요 채널 식별 → 채널 스케일링 → INT4 양자화 → 저장·평가`

핵심 설정은 GPTQ와 동일: **비트 수(4bit)**, **group_size(128)**, **calibration 데이터**(도메인 일치 중요)

### 코드

```python
from gptqmodel import GPTQModel, AWQConfig
from datasets import load_dataset

raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
calibration_dataset = [t for t in raw["text"] if t.strip()][:256]

quant_config = AWQConfig(bits=4, group_size=128)   # format="gemm", 비대칭 기본
model = GPTQModel.load("Qwen/Qwen2.5-0.5B-Instruct", quant_config)
model.quantize(calibration_dataset, batch_size=1)   # 활성화 통계 → 채널 스케일 탐색 → INT4
model.save("qwen2-5-0-5b-awq-int4")
# 0.92GB → 0.43GB (약 53.6% 감소)
```

```python
# 이미 양자화된 공개 AWQ 모델 로드 (T4 가능)
from gptqmodel import GPTQModel, BACKEND
model = GPTQModel.load("Qwen/Qwen2.5-0.5B-Instruct-AWQ", backend=BACKEND.TORCH, device="cuda:0")
```

> **직접 양자화는 A100(sm_80)+ 필요** — T4(sm_75)는 `cp.async requires sm_80` 오류. 로드·추론만은 T4 가능. GPTQ와 진입점(`GPTQModel`) 동일, 설정 객체만 `AWQConfig`로 다름

> 논문: *AWQ* (Lin et al., 2023) — **MLSys 2024 Best Paper Award**

---

## AWQ vs GPTQ — 사전 보호 vs 사후 보상

둘 다 weight-only PTQ지만 **오차를 줄이는 접근 방향**이 반대

| 항목 | **AWQ** | **GPTQ** |
|---|---|---|
| 핵심 전략 | 활성화 통계 기반 **채널 스케일링** | Hessian 기반 **오차 보상** |
| 오차 처리 | 양자화 **전에** 중요 채널 보호 | 양자화 **후** 생긴 오차를 보정 |
| 판단 기준 | activation statistics | 레이어 재구성 오차의 2차 정보 |
| 수학 도구 | 채널별 스케일 탐색, 등가 변환 | damped inverse Hessian |
| 양자화 시간 | 가벼운 편 | 무거운 편 |
| 실무 장점 | 흐름 단순, 서빙 적용성 좋음 | 체크포인트·생태계 넓음 |
| 대표 관점 | **사전 보호** | **사후 보상** |

> 둘 다 calibration 필요. 4bit·g128 정확도는 비슷하며 조건에 따라 AWQ가 소폭 우수하다는 보고도 있음 — **어느 쪽이 항상 낫다고 단정 불가**

### AWQ의 한계

- **Calibration 도메인 불일치**: 영어 웹문서로 calibration하고 한국어 상담·코드·의료에 쓰면, 찾은 중요 채널이 실제 입력에 최적이 아닐 수 있음
- **채널 단위 보호의 한계**: 같은 채널 안에서 특정 가중치만 유난히 민감하면 채널 스케일 하나로 부족 → 개별 가중치 수준으로 보정하는 GPTQ가 유리할 수 있음

> 양자화 후에는 반드시 원본과 **같은 조건에서 비교** (Perplexity·벤치마크·응답 품질·메모리·속도) — 다음 절 Quantization Evaluation

---

## Quantization Evaluation

양자화 전후의 **모델 품질**과 **서빙 효율**을 같은 조건에서 비교해, 배포해도 되는지 판단하는 과정

> "메모리가 줄었다"만으로 판단하면 안 됨 — 품질 손실이 얼마인지, 실제 서빙에서 속도·메모리 이점이 있는지 **함께** 확인

### 두 영역으로 나눠 평가

품질이 유지돼도 서빙 이점이 작을 수 있고, 반대로 속도·메모리 이점은 커도 품질 손실이 허용 범위를 넘을 수 있음

| 영역 | 지표 |
|---|---|
| **모델 품질** | Perplexity(PPL), Task Benchmark, Human Evaluation |
| **서빙 성능** | TTFT, ITL, E2E Latency, Throughput, Peak VRAM, 파일 크기 |

> 같은 양자화 모델이라도 백엔드(vLLM·TensorRT-LLM·llama.cpp·exllama)가 다르면 서빙 성능이 크게 달라짐 → **품질과 서빙은 분리해서 평가**

### 모델 품질 지표

| 지표 | 설명 |
|---|---|
| **Perplexity (PPL)** | 다음 토큰 예측 품질, **낮을수록 좋음**. 자동 계산 빨라 **1차 지표**로 사용 |
| **Task Benchmark** | MMLU(종합 지식) · GSM8K(수학 추론) · HumanEval(코드 생성) 등 작업별 정확도 |
| **Human Evaluation** | 사람이 두 응답을 비교. 말투·일관성·설명력 등 자동 지표로 못 잡는 품질 |

> PPL만으론 부족 — 수학·코드·전문 지식은 **별도 Task Benchmark 필요**

### 서빙 성능 지표

| 지표 | 의미 | 관련 |
|---|---|---|
| **TTFT** (Time To First Token) | 첫 토큰까지 걸리는 시간 | 사용자 체감 |
| **ITL** (Inter-Token Latency) | 토큰 간 평균 지연 | 사용자 체감 |
| **E2E Latency** | 요청~응답 완료 전체 시간 | 사용자 체감 |
| **Throughput** | 초당 처리 토큰/요청 수 | 운영 비용 |
| **Peak VRAM** | 실행 중 GPU 메모리 정점 | 운영 비용 |
| 모델 파일 크기 | 디스크 저장 크기 | 다운로드·배포 비용 |

> 서빙 성능은 모델만으로 결정되지 않음 — 백엔드·동시성·배치·입출력 길이·GPU 종류에 좌우

### 판단 기준 (예시 — 절대 합격선 아님)

| 항목 | 일반 서비스 | 민감 서비스(코드·금융·의료) |
|---|---|---|
| PPL 증가 | `+0.5 ~ +1.0` 이내 | `+0.5` 이내 |
| 작업 정확도 하락 | `-1~3%` 이내 | `-1%` 이내 |
| Peak VRAM 감소 | 최소 `20~30%` | 품질 유지 우선 |
| Throughput | FP16 대비 같거나 향상 | 품질 유지 범위에서만 인정 |

> **허용 기준은 평가 시작 전에 정하기** — 측정값 보고 기준을 맞추면 결과에 끼워 맞추게 됨

### 평가 흐름

`① 기준(FP16) 측정 → ② 양자화 모델 준비 → ③ 통제 조건 고정 → ④ 양자화 모델 측정 → ⑤ 비교 → ⑥ 배포 판단`

> **통제 조건**: 평가 데이터셋·토크나이저·프롬프트 형식·입출력 길이·배치·동시성·하드웨어·백엔드를 **모두 동일하게**. 안 그러면 양자화 효과인지 환경 차이인지 구분 불가

### 코드 (PPL 비교)

```python
@torch.no_grad()
def evaluate_ppl(hf_model, tokenizer, max_length=2048, stride=512, max_eval_tokens=20480):
    hf_model.eval()
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids_all = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids[:, :max_eval_tokens]
    nlls, prev_end = [], 0
    for begin in range(0, ids_all.size(1), stride):          # 슬라이딩 윈도우
        end = min(begin + max_length, ids_all.size(1))
        trg_len = end - prev_end
        ids = ids_all[:, begin:end].to(hf_model.device)
        target = ids.clone(); target[:, :-trg_len] = -100     # 이전 문맥은 손실에서 제외
        nlls.append(hf_model(input_ids=ids, labels=target).loss.float() * trg_len)
        prev_end = end
        if end == ids_all.size(1): break
    return torch.exp(torch.stack(nlls).sum() / prev_end).item()
```

> **핵심**: FP16과 INT4를 **같은 함수·같은 데이터셋·같은 윈도우 설정**으로 평가해야 PPL 차이를 양자화 효과로 해석 가능

### 실측 예시 (Qwen2.5-0.5B, T4)

| | PPL | Peak VRAM | disk |
|---|---|---|---|
| FP16 | 11.437 | 3950MB | 942MB |
| GPTQ INT4 | 13.190 | 3446MB | 438MB |
| **차이** | **+1.753** | -504MB | **-54%** |

> **주의**: disk는 54% 줄었지만 **VRAM은 조금만 감소** — weight-only 양자화는 가중치만 줄이고 **KV cache·activation·runtime buffer는 그대로** 남기 때문. "INT4니까 VRAM도 1/4"는 오해

> **4bit가 항상 빠른 것도 아님** — 저비트 가중치를 다시 푸는(역양자화) 비용이 생기거나 GPU·백엔드에 맞는 최적화 커널이 없으면 **오히려 느려질 수 있음** → 반드시 실제 환경에서 측정

### lm-evaluation-harness

EleutherAI의 LLM 평가 프레임워크 — 여러 모델을 **같은 프롬프트·태스크·채점 방식**으로 평가 (Hugging Face Open LLM Leaderboard에서도 사용)

> 양자화 전후 비교에서 유용한 이유: **평가 절차를 고정**해줌. 사람이 매번 프롬프트·few-shot 개수·채점 방식을 맞추면 차이가 생김. MMLU·GSM8K·HumanEval 등 서비스 목적에 가까운 태스크를 같은 조건으로 비교
