# 6-1. PEFT 기법 (PEFT · LoRA · QLoRA · Adapter Merge · Model Merging)

사전학습 모델의 **대부분 파라미터를 고정**하고, 소수의 추가 파라미터만 조정해 특정 작업에 적응시키는 파인튜닝 방법들의 **총칭**

> 3축 중 **효율적 재학습** — 모델 전체를 다시 학습하지 않고, 작은 학습 부품만 붙여 적응

---

## PEFT (Parameter-Efficient Fine-Tuning)

단일 기법이 아니라 **계열 전체**를 가리키는 상위 개념

> 비유: Full Fine-Tuning은 애플리케이션 **전체 코드를 작업별로 복사·수정**하는 방식, PEFT는 핵심 코드는 그대로 두고 **플러그인만 갈아 끼우는** 방식

### PEFT vs Full Fine-Tuning

| 항목 | Full Fine-Tuning | **PEFT** |
|---|---|---|
| 학습 대상 | 모든 파라미터 (100%) | 소수의 추가 파라미터 (**대체로 1% 미만**) |
| 원본 가중치 | 수정됨 | **대부분 고정** |
| GPU 메모리 | 큼 | 상대적으로 작음 |
| 저장 공간 | 작업마다 **모델 전체 사본** | 작업마다 **어댑터만** |
| 멀티태스크 | 작업별 모델 사본 관리 | **작업별 어댑터 교체** |
| 성능 | 가장 높음 | 일부 작업에서 근접 |

> Full FT는 가중치뿐 아니라 **기울기·옵티마이저 상태도 전체 파라미터에 대해** 필요 → 모델이 커질수록 메모리 급증. PEFT는 작은 학습 대상에 대해서만 필요

### 어댑터 (Adapter)

원본 가중치는 그대로 두고, 작업에 맞게 **추가로 붙이는 작은 학습 부품**

> 같은 기반 모델 위에 "고객 응대용 어댑터", "요약용 어댑터"를 따로 학습해 **교체하며 사용**

### 기법 분류

| 갈래 | 대표 기법 | 설명 |
|---|---|---|
| **어댑터·가중치 수정** | **LoRA**, IA3, AdaLoRA, Adapter Layers | 모델 내부에 작은 학습 파라미터 추가 |
| **양자화 결합** | **QLoRA** | 양자화된 기반 모델 위에 LoRA 어댑터를 올려 학습 |
| **병합·결합** | **Adapter Merge**, **Model Merging** | 학습된 어댑터·여러 모델의 가중치를 결합 |
| 프롬프트 기반 | Prompt Tuning, Prefix Tuning | 가중치 대신 학습 가능한 프롬프트·prefix 추가 |

### 사용 이유

- **메모리·시간·저장 절감**: 원본 유지 + 작은 파라미터만 업데이트
- **하나의 기반 모델 재사용**: 고객 응대·요약·번역을 모델 사본 없이 어댑터 교체로 관리
- **실험 속도**: 데이터셋·하이퍼파라미터 바꿔가며 빠르게 반복 (전체 모델 사본 불필요)

> **한계**: 작업이 사전학습 지식과 크게 다르거나 **새 지식을 대량 주입**해야 하면 작은 조정 파라미터로 부족할 수 있음. 메모리·성능 이득은 고정 비율이 아니라 모델 크기·정밀도·시퀀스 길이·배치·학습 대상 위치에 좌우 → **같은 조건에서 비교 측정 필요**

### 사용 방법

| 순서 | 단계 |
|---|---|
| 1 | 사전학습 기반 모델 로드 |
| 2 | **원본 가중치 고정** (학습 대상에서 제외) |
| 3 | PEFT 기법 선택 |
| 4 | 학습 파라미터 추가 (어댑터·저랭크 행렬·프롬프트) |
| 5 | **추가된 파라미터만** 업데이트 |
| 6 | **PEFT 결과물만** 저장 (원본 전체 X) |
| 7 | 기반 모델 위에 올려 추론, 또는 원본에 **병합**해 배포 |

> 원본 가중치는 **순전파에선 계속 사용**되지만 학습 대상에선 제외 — 기울기가 흐르지 않음

### 코드 (Hugging Face `peft` + `trl`)

- **`peft`**: LoRA·QLoRA·Prompt Tuning 등을 모델에 붙이고 어댑터만 학습
- **`trl`** (Transformer Reinforcement Learning): SFT·DPO 등 LLM 후학습
  - **SFT**: 프롬프트-정답 쌍으로 답변 형식을 따라 하도록 학습
  - **DPO**: 선호/비선호 응답 비교 데이터로 더 좋은 응답을 고르도록 학습

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                                  # 저랭크 차원 (클수록 학습 파라미터↑)
    lora_alpha=16,                        # 스케일링 팩터
    target_modules=["q_proj", "v_proj"],  # LoRA를 붙일 레이어 (Attention Q·V 투영층)
    lora_dropout=0.05,                    # 과적합 방지
    bias="none",                          # bias는 학습 안 함
    task_type="CAUSAL_LM",                # 다음 토큰 예측 작업
)
```

학습 후 저장되는 파일:
```
adapter_model.safetensors   ← 학습된 어댑터 가중치 (원본 모델 전체 X)
adapter_config.json         ← 어댑터 설정
```

> **핵심**: 원본 모델 전체를 다시 저장하는 게 아니라 **어댑터 파일만** 저장. `push_to_hub()`로 어댑터만 공유 가능 (크기가 작아 빠름)

---

## PEFT vs Full Fine-Tuning vs Prompt Engineering

| 항목 | **PEFT** | Full Fine-Tuning | Prompt Engineering |
|---|---|---|---|
| 가중치 수정 | 추가 파라미터만 | 전체 | 없음 |
| 학습 | 필요 (대상 작음) | 필요 (전체) | **불필요** |
| GPU 메모리 | 중간 | 큼 | 추론용만 |
| 저장 | 작업별 어댑터 | 작업별 모델 전체 사본 | 프롬프트 텍스트 |
| 도메인 적응 | 높음 | **가장 높음** | 낮음~중간 |
| 데이터 필요량 | 수백~수만 건 | 더 많이 필요 | 불필요 |
| 과적합 위험 | 비교적 낮음 | 소량 데이터에서 높을 수 있음 | 거의 없음 |
| 적합 상황 | **제한된 GPU로 작업 적응** | 자원 충분 + 최고 성능 | 학습 없이 빠른 테스트 |

> 선택 기준은 **자원과 목표** — 빠른 시도는 Prompt Engineering, 최고 성능·자원 충분하면 Full FT, **제한된 GPU에서 실제 작업 적응**이 필요하면 PEFT

---

## LoRA (Low-Rank Adaptation)

원본 가중치 `W`는 **고정**하고, 작업 적응에 필요한 가중치 **변화량 `ΔW`를 두 개의 저랭크 행렬 곱으로 근사**해 소수 파라미터만 학습하는 PEFT 기법

> **`ΔW`**: 파인튜닝으로 원본 `W`에 더해지는 변화량 (작업 적응에 필요한 수정분)

> **PEFT ↔ LoRA 관계**: PEFT는 "모델 전체를 안 건드리고 가볍게 적응시키자"는 **범주**, LoRA는 "원본은 고정하고 **작은 저랭크 행렬 두 개만** 학습하자"고 답하는 **구체 기법**

### 원리 — 변화량만 작게 학습

Full FT는 기존 가중치 전체를 직접 수정 (설정 파일 전체를 새로 만드는 격)
LoRA는 원본을 두고 **"이 작업엔 어느 방향으로 얼마나 바뀌어야 하나"** 만 작은 어댑터로 학습 (변경분만 담은 **패치 파일**)

변화량을 큰 덩어리로 학습하지 않고 **두 행렬로 분해**:
- 하나는 입력을 **작은 차원으로 압축**
- 다른 하나는 다시 **원래 출력 차원으로 확장**

> 이 작은 통로의 크기가 **`rank`** — 작으면 가볍지만 표현 가능한 변화가 제한, 크면 복잡한 변화 표현 가능하나 파라미터·메모리↑

### 학습 파라미터가 줄어드는 이유

> **학습 파라미터** = 학습 중 실제로 값이 바뀌는 가중치. 줄어들면 **기울기·옵티마이저 상태도 적게** 필요 → GPU 메모리↓

`4,096 × 4,096` 가중치에 `rank=16` 적용 시:

```
Full FT: 4,096 × 4,096 전체 변화량 학습
LoRA:    A (16 × 4,096)  +  B (4,096 × 16) 만 학습
```

| | Full Fine-Tuning | LoRA (`rank=16`) |
|---|---|---|
| 학습 파라미터 수 | `16,777,216` | `131,072` |
| 비율 | 100% | **약 0.78%** |

> 원본 모델이 사라지는 게 아님 — 원본은 그대로 쓰고 **작은 수정분만 추가 학습**

### 순전파 흐름

```
입력 ─┬─→ 원본 가중치 경로 (고정, 학습 안 함) ─┐
      │                                        ＋ ─→ 최종 출력
      └─→ LoRA 어댑터 경로 (학습)  ───────────┘
```

- **원본 경로**: 사전학습 지식 사용, 학습 중 가중치 안 바뀜
- **LoRA 경로**: "현재 작업에 맞게 얼마나 보정할지" 계산 → 원본 출력에 **더할 작은 수정값** 생성

> LoRA는 원본 모델을 **대체하지 않고 얹음** — 기존 API 로직을 고치는 게 아니라 **미들웨어를 붙여 일부 동작만 보정**하는 구조

### 초기화 — 시작 시점 영향 0

어댑터를 붙이자마자 출력이 크게 달라지면 사전학습 지식을 잃은 채 학습을 시작하게 됨 → 시작 시 **영향이 0이 되도록** 초기화

| 행렬 | 초기화 | 의미 |
|---|---|---|
| `A` | 무작위 초기화 | 학습이 진행될 출발점 제공 |
| `B` | **0으로 초기화** | 어댑터 출력이 처음엔 0 → 원본과 동일한 출력 |

> `B=0`이므로 어댑터 경로의 결과도 0 → **학습 시작 시점 출력 = 원본 모델 출력**. 학습이 진행되며 `A`·`B`가 함께 업데이트되고 어댑터가 점차 보정 역할 수행

> 비유: 새 플러그인을 설치했지만 **처음엔 꺼져 있는 상태** — 학습하며 점점 켜짐

### 스케일링 — `lora_alpha`

어댑터가 만든 보정값을 **그대로 더하지 않고, 어느 정도 크기로 반영할지 조절**하는 값

| `lora_alpha` | 결과 |
|---|---|
| 너무 작음 | 어댑터가 학습한 변화가 약하게 반영 → **새 작업에 충분히 적응 못 함** |
| 너무 큼 | 어댑터 영향이 과해 **학습 불안정** / 원본 모델의 기존 능력을 지나치게 흔듦 |

> 어댑터의 **"출력 볼륨"** 조절값. 단독으로 보지 않고 **`rank`와 함께** — 실제 반영 강도는 **`lora_alpha / rank` 비율**로 정해짐

| rank | lora_alpha | 용도 |
|---|---|---|
| `8` | `16` | 가벼운 기본 설정 |
| `16` | `32` | **자주 쓰는 시작 설정** |
| `32` | `64` | 더 큰 표현력 시도 |

> 관례적으로 **`lora_alpha = 2 × rank`** 로 시작. 고정 정답 아님 — 데이터셋 크기·작업 난이도·모델 크기·학습률·`target_modules`에 따라 달라짐

### 사용 이유

- 학습 파라미터↓ → **기울기·옵티마이저 상태 메모리도 함께↓**
- **하나의 기반 모델 + 작업별 어댑터**: 고객사·도메인·작업별로 어댑터만 관리 (모델 사본 불필요)
- 학습 후 **원본에 병합 가능** → 추론 시 일반 모델처럼 실행 (단일 작업 배포 시 운영 단순화)

> 논문에서도 학습 파라미터·GPU 메모리를 크게 줄이며 Full FT에 근접한 성능을 보고. 단, **모든 모델·작업에서 보장되진 않음** — 모델 크기·데이터셋·rank·적용 레이어·학습 조건에 좌우

### `target_modules` — 어디에 붙일지

LoRA 어댑터를 **모델의 어느 레이어에 붙일지** 지정

> 전체 서비스가 아니라 **특정 함수·모듈에만 플러그인**을 붙이는 것. 많이 붙일수록 표현력↑ 파라미터·메모리↑

### 사용 방법

`LoraConfig 설정 → get_peft_model(어댑터 부착 + 원본 자동 freeze) → 어댑터만 학습 → save_pretrained → PeftModel.from_pretrained 또는 merge_and_unload`

| 함수 | 역할 |
|---|---|
| `LoraConfig` | 어댑터 구조(rank·스케일링·적용 모듈) 정의 |
| `get_peft_model` | 어댑터 부착 + **원본 가중치 자동 고정(freeze)** |
| `SFTConfig` / `SFTTrainer` (TRL) | Supervised Fine-Tuning 학습 설정·학습기 |
| `PeftModel.from_pretrained` | 추론 시 기반 모델 위에 어댑터 재부착 |
| `merge_and_unload` | 어댑터를 원본에 **흡수해 단일 모델화** (추론 시 추가 연산 없음) |

> **SFT**: 정답 있는 예시 데이터("질문 → 좋은 답변")로 답변 형식을 따라 하도록 학습 / **instruction tuning**: SFT로 지시 수행 능력을 높이는 학습 목적

### LoraConfig 인자

| 인자 | 역할 | 예시 |
|---|---|---|
| `r` | 저랭크 차원. 표현력·파라미터 결정 | `8`, `16`, `32` |
| `lora_alpha` | 스케일링. 반영 강도 = `lora_alpha / r` | `16`, `32` |
| `target_modules` | LoRA를 붙일 모듈 | `["q_proj","v_proj"]`, `"all-linear"` |
| `lora_dropout` | 어댑터 경로 드롭아웃 (과적합 방지) | `0.05`, `0.1` |
| `bias` | bias 학습 여부 | `"none"` (보통) |
| `use_rslora` | Rank-Stabilized LoRA. 큰 rank에서 학습 안정화 | `False`, `True` |
| `task_type` | 작업 유형 | `"CAUSAL_LM"`, `"SEQ_CLS"` |

> **CAUSAL_LM**: 앞 토큰으로 다음 토큰 예측 (텍스트 생성) / **SEQ_CLS**: 문장 전체를 보고 라벨 하나 예측 (분류)

> 가장 중요한 3개는 **`r`, `lora_alpha`, `target_modules`**. 처음엔 `r=8, lora_alpha=16, target_modules=["q_proj","v_proj"]`로 가볍게 시작

### 코드

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token      # Causal LM 패딩용

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],   # Attention Q·K·V·O
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)    # 원본 자동 freeze
model.print_trainable_parameters()
# trainable params: 4,358,144 || all params: 1,548,072,448 || trainable%: 0.2815
```

```python
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

dataset = load_dataset("trl-lib/Capybara", split="train[:200]")
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./lora-output", max_steps=20,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=2e-4,        # LoRA는 학습 대상이 적어 Full FT보다 큰 학습률
        fp16=True,                 # T4는 BF16 미지원 (A100이면 bf16=True)
        report_to="none",
    ),
    train_dataset=dataset, processing_class=tokenizer,
)
trainer.train()      # 어댑터 파라미터만 학습
```

```python
from peft import PeftModel

model.save_pretrained("./lora_adapter")     # 어댑터만 저장 (원본 가중치 X)

# 추론: 기반 모델 다시 로드 → 어댑터 부착
base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
inference_model = PeftModel.from_pretrained(base_model, "./lora_adapter")
```

> **LoRA 어댑터는 단독으로 동작하지 않음** — 반드시 학습 때 쓴 **기반 모델 위에 다시 올려야** 함. Hub 어댑터도 동일: 공개자가 명시한 기반 모델 ID를 확인 후 `PeftModel.from_pretrained(base_model, "<리포 ID>")`

### LoRA vs Full Fine-Tuning

| 항목 | **LoRA** | Full Fine-Tuning |
|---|---|---|
| 학습 대상 | 어댑터 파라미터 | 모델 전체 |
| 원본 가중치 | 고정 | 직접 수정 |
| 학습 파라미터 비율 | 대체로 **1% 미만** | 100% |
| 저장 | 작업별 어댑터만 | 작업별 모델 전체 |
| 추론 시 병합 | 가능 | 개념 없음 |
| 성능 | Full FT에 근접한 사례 많음 | **성능 상한 가장 높음** |

### rank가 커지면

학습 파라미터 수는 **대체로 rank에 비례** → rank 2배면 파라미터도 거의 2배 → 기울기·옵티마이저 상태 메모리도 증가

| rank | 표현력 | 메모리 | 적합 작업 |
|---|---|---|---|
| `4` | 낮음 | 작음 | 단순 형식 맞춤, 짧은 분류 |
| `8` | 보통 | 작음 | **instruction tuning 시작점** |
| `16` | 높음 | 중간 | 범용 tuning, 가벼운 도메인 적응 |
| `32` | 더 높음 | 큼 | 도메인 적응, 말투·형식 학습 |
| `64` | 매우 높음 | 매우 큼 | 큰 도메인 전환 |

> rank는 **`target_modules`와 함께** 결정 — `q_proj,v_proj`만 vs `"all-linear"`는 같은 rank라도 파라미터 수가 크게 다름. 실무: `r=8` 또는 `16`으로 기준선 → 부족하면 rank↑ 또는 `target_modules` 확장 / 과적합이면 rank↓ 또는 `lora_dropout`↑

### 한계

- **기반 모델 의존**: 어댑터는 학습 당시 기반 모델 가중치 기준으로 만들어짐 → **모델 버전이 바뀌면 의도대로 동작 안 할 수 있음** (프레임워크 버전 다른 플러그인)
- **`target_modules` 밖의 변화는 못 배움**: 필요한 변화가 어댑터를 안 붙인 영역에 많으면 성능 부족 → 범위 확장·rank 조정 검토
- **어댑터 합성 보장 X**: 같은 기반 모델이라도 "법률 요약용" + "친근한 말투용"을 함께 적용한다고 두 능력이 안정적으로 합쳐지지 않음 → **별도 검증 필수**

> 공개 어댑터 사용 시 **기반 모델 ID·`adapter_config.json`의 `target_modules`·라이선스**를 함께 확인

> 논문: *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., Microsoft, 2021, **ICLR 2022**)

---

## QLoRA (Quantized LoRA)

**4-bit NF4로 양자화된 기반 모델은 고정**하고, **16-bit LoRA 어댑터만 학습**해 메모리 사용량을 줄이는 PEFT 기법

> LoRA 구조는 그대로 — 차이는 **기반 모델을 16-bit가 아니라 4-bit로 로드**한다는 것
> 비유: LoRA는 서비스에 플러그인만 붙임 / QLoRA는 **서비스 패키지 자체를 압축해 메모리에 올리고** 플러그인만 학습

### 핵심 구성 요소 — 각자 줄이는 메모리 위치가 다름

| 구성 요소 | 적용 대상 | 줄이는 지점 |
|---|---|---|
| **NF4** (NormalFloat 4-bit) | 기반 모델 가중치 | 기반 모델 메모리 |
| **Double Quantization** | 양자화 상수(scale) | 양자화 부가 메모리 (파라미터당 ~0.37비트 추가 절감) |
| **Paged Optimizer** | 옵티마이저 상태 | GPU 부족 시 CPU RAM으로 옮겼다 가져옴 (OOM 방지) |
| **LoRA 어댑터** | 학습 대상 | 학습 파라미터·기울기·옵티마이저 부담 |

### 메모리 구성 — 이점은 어디서 오나

| 7B 모델 기반 가중치 | 메모리 |
|---|---|
| FP16/BF16 LoRA (2B/param) | 약 14GB |
| **QLoRA 4-bit** (0.5B/param) | 약 **3.5GB** (+ 양자화 상수) |

> **핵심**: QLoRA의 메모리 이점은 **어댑터가 작아져서가 아님** — 고정된 기반 모델을 16-bit가 아니라 4-bit로 올리기 때문. 모델이 클수록 절감 효과↑

### NF4 — 균등하지 않은 4비트

일반 INT4는 범위를 **같은 간격**으로 16등분. NF4는 사전학습 가중치가 **0 중심 정규분포**에 가깝다는 관찰을 반영해 **비균등 배치**

```
INT4: |----|----|----|----|----|----|----|----|     같은 간격
      -1.0                                +1.0

NF4:  |------|-----|---|--|-|-|-|--|---|-----|------|
      -1.0     넓음      0 촘촘함      넓음     +1.0
```

| | 일반 INT4 | **NF4** |
|---|---|---|
| 구간 배치 | 같은 간격 | 값이 몰리는 0 근처를 **촘촘하게** |
| 장점 | 단순 | 정규분포형 가중치에서 **정보 손실↓** |

> 비유: 모든 API에 같은 캐시 정책 vs **트래픽 많은 경로에 더 촘촘한 정책**

> **왜 어댑터만 16-bit?** 기반 모델은 학습 안 하니 4-bit로 압축해도 됨. 어댑터는 실제 학습되므로 **작은 변화도 안정적으로 누적**하려면 16-bit 필요

### 사용 방법 — LoRA와의 차이는 앞 3단계

| 순서 | 단계 | LoRA와 차이 |
|---|---|---|
| 1 | `BitsAndBytesConfig` 설정 | **QLoRA 전용** |
| 2 | 4-bit NF4로 모델 로드 | **QLoRA 전용** |
| 3 | `prepare_model_for_kbit_training()` | **QLoRA 전용** (4-bit 위에서 gradient가 올바르게 흐르도록 준비) |
| 4~8 | LoraConfig → get_peft_model → 학습 → 저장 → 배포 | LoRA와 동일 |

| `BitsAndBytesConfig` 인자 | 역할 |
|---|---|
| `load_in_4bit=True` | 4-bit 로드 (**주 메모리 절감**) |
| `bnb_4bit_quant_type="nf4"` | NF4 선택 (**미지정 시 기본 `"fp4"`** → 정규분포 이점 못 받음) |
| `bnb_4bit_use_double_quant=True` | 양자화 상수 한 번 더 압축 |
| `bnb_4bit_compute_dtype` | 4-bit 저장이어도 **실제 행렬곱은 이 dtype** (`torch.bfloat16` 등) |

### 코드

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", quantization_config=bnb_config, device_map="auto",
)
model = prepare_model_for_kbit_training(model)   # ← QLoRA 특유 단계

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules="all-linear",     # QLoRA 논문 흐름에서 자주 사용 (16-bit FT에 근접시키려)
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 18,464,768 || all params: 1,562,179,072 || trainable%: 1.1820
```

> `trainable% 1.18`은 `all-linear`로 모든 선형층에 붙였기 때문 (LoRA 예시의 `q,k,v,o`만은 0.28%). 메모리 부족하면 `["q_proj","k_proj","v_proj","o_proj"]`로 축소

```python
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./qlora-output", max_steps=30,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=2e-4, bf16=True, max_length=1024,
        gradient_checkpointing=True,                             # 활성화 메모리 절약
        gradient_checkpointing_kwargs={"use_reentrant": False},   # k-bit 학습 시 권장
        optim="paged_adamw_8bit",                                # ← Paged Optimizer (OOM 방지)
    ),
    train_dataset=dataset, processing_class=tokenizer,
)
trainer.train()
model.save_pretrained("qwen2-5-1-5b-qlora")   # 어댑터만 저장
```

```python
# 추론: 같은 NF4 설정으로 기반 모델 로드 → 어댑터 부착 (4-bit base + 16-bit adapter)
base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(base_model, "qwen2-5-1-5b-qlora")
```

> OOM 시 우선순위: `per_device_train_batch_size` ↓ → `max_length` ↓ → `target_modules` 범위 ↓ → gradient checkpointing·Paged Optimizer 확인

### QLoRA vs LoRA

| 항목 | **QLoRA** | FP16/BF16 LoRA |
|---|---|---|
| **기반 모델 정밀도** | **4-bit NF4** | FP16/BF16 |
| 어댑터 정밀도 | FP16/BF16 | FP16/BF16 |
| 학습 대상 | 어댑터만 | 어댑터만 (동일) |
| GPU 메모리 | 더 작음 | 상대적으로 큼 |
| 학습 속도 | **역양자화 때문에 약간 느림** | 빠름 |
| 추가 라이브러리 | `bitsandbytes` 필요 | `peft` 중심 |
| 적합 상황 | GPU 부족·더 큰 모델 | 메모리 여유·속도 중요 |

> 실무 흐름: **FP16/BF16 LoRA 먼저 시도 → GPU 메모리 부족하거나 더 큰 모델이면 QLoRA**

> 성능은 논문에서 16-bit LoRA·Full FT에 근접 보고. 단, 4-bit 기반이라 일부 작업에서 차이 날 수 있어 **프로젝트별 검증 필요**

> 논문: *QLoRA: Efficient Finetuning of Quantized LLMs* (Dettmers et al., 2023, **NeurIPS 2023**)

---

## Adapter Merge

LoRA처럼 **가중치 변화량으로 표현되는 어댑터를 원본 가중치에 합산**해, 별도 어댑터 없이 동작하는 **단일 모델 체크포인트**로 만드는 과정

> 어댑터를 "붙여서 쓰는 상태" → 변화량을 미리 더해 **"하나의 모델로 굳히는 상태"**

### 병합의 원리

$$W' = W + \Delta W$$

```
병합 전:  출력 = 원본 모델 계산  +  어댑터 보정값     (매 추론마다 두 경로 계산)
병합 후:  출력 = 병합된 단일 모델 계산               (어댑터 구조 제거)
```

> **새로운 학습이 아님** — 이미 학습된 변화량을 원본에 더해 저장하는 **사전 계산**. 비유: 플러그인의 변경 사항을 본체 코드에 미리 반영해 **단일 빌드**로 만드는 것

> ⚠️ **병합된 체크포인트만으로는 어댑터를 다시 분리할 수 없음** → 원본 모델과 어댑터 파일은 **따로 백업**

### 어댑터 유지 vs 병합

| | 어댑터 유지 | **병합** |
|---|---|---|
| 적합 | 실험 단계, **작업별 어댑터 교체** | 단일 작업 안정 배포 |
| 로드 | 기반 모델 + 어댑터 함께 | 단일 모델 |
| 추론 환경 | PEFT 어댑터 로드 지원 필요 | **일반 모델처럼 사용** |

### 사용 이유

- 배포 시 **어댑터 파일 관리 부담·호환성 문제 감소** — PEFT 모르는 추론 코드에서도 사용 가능
- 이후 **GPTQ·AWQ 서빙용 양자화 파이프라인**으로 넘기기 쉬움

> **핵심 이점은 "속도"가 아님** — 어댑터 경로를 제거해 구조는 단순해지지만, 실제 속도 차는 추론 엔진·배치·시퀀스 길이·양자화 방식·멀티-LoRA 지원 여부에 좌우. **배포 구조 단순화 + 일반 추론 환경 호환성**이 진짜 이점

### 사용 방법 — `merge_and_unload()`

| 순서 | 단계 |
|---|---|
| 1 | 기반 모델을 **FP16/BF16으로 로드** (학습이 4-bit였어도) |
| 2 | `PeftModel.from_pretrained(base_model, adapter_path)` |
| 3 | `merge_and_unload(safe_merge=True)` — 합산 후 어댑터 구조 제거 |
| 4 | `save_pretrained(..., safe_serialization=True)` + 토크나이저 저장 |
| 5 | 추론 또는 GPTQ/AWQ 재양자화 |

| 인자 | 역할 |
|---|---|
| `safe_merge=True` | 병합 중 **NaN/Inf 감지 시 중단** (모델 손상 방지) — 권장 |
| `safe_serialization=True` | pickle `.bin` 대신 **safetensors**로 저장 — 권장 |
| `progressbar` | 병합 진행 상황 표시 (대형 모델) |
| `adapter_names` | 여러 어댑터 중 병합할 것 선택 |

### 코드

```python
import torch
from peft import PeftModel

# 병합은 4-bit가 아니라 BF16으로 재로드 (학습 정밀도와 맞춤)
base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base_model, "qwen2-5-1-5b-qlora")

merged_model = model.merge_and_unload(safe_merge=True)   # W' = W + ΔW, 어댑터 제거

merged_model.save_pretrained("qwen2-5-1-5b-qlora-merged", safe_serialization=True)
tokenizer.save_pretrained("qwen2-5-1-5b-qlora-merged")   # 토크나이저도 함께 저장
```

> ⚠️ **QLoRA로 4-bit 학습했어도 병합은 BF16/FP16에서** — 4-bit 격자 위에 어댑터 변화량을 반영하면 **반올림 오차**로 병합 전후 출력이 달라질 수 있음

> **권장 흐름**: `4-bit로 학습 → 어댑터 저장 → BF16/FP16 재로드 → 병합 → 필요 시 GPTQ/AWQ 재양자화`
> (학습은 QLoRA의 메모리 효율, 배포는 병합 모델 + 서빙용 양자화)

---

## Model Merging

동일하거나 호환 가능한 아키텍처의 **여러 모델 체크포인트를 가중치 공간에서 결합**해, **추가 학습 없이** 하나의 새 체크포인트를 만드는 기법

> **"가중치 공간에서 결합"** = 같은 구조의 모델끼리 **같은 위치의 가중치**를 찾아 평균·보간·task vector 결합으로 새 값 계산

> 앞선 양자화·PEFT는 **하나의 모델을 가볍게** 하는 방법 / Model Merging은 **여러 모델의 강점을 하나로 모으는** 방법

### 전제 조건 — 아무 모델이나 못 섞음

레이어 수, hidden size, attention 구조, **tokenizer·chat template**이 크게 다르면 병합이 어렵거나 품질 급락

> 비유: 같은 코드베이스에서 갈라진 브랜치는 merge 가능하지만, 전혀 다른 프레임워크 프로젝트는 파일을 합칠 수 없음. **입문은 동일 기반 모델에서 파인튜닝된 모델끼리**

### 병합 전략

| 전략 | 방식 | base_model 필요 |
|---|---|---|
| **Linear** | 같은 위치 가중치를 정해진 비율로 **단순 평균** | 불필요 |
| **SLERP** | 직선 평균이 아닌 **구면 보간** — 가중치 **방향성 보존** | 불필요 |
| **Task Arithmetic** | 기반 모델 대비 **변화량(task vector)** 을 더함 | **필요** |
| **TIES** | 변화량 중 작은 값 제거 + **충돌하는 방향 정리** | **필요** |
| **DARE** | 변화량 일부를 **드롭 후 재조정** (과도한 누적 방지) | **필요** |

> 입문은 **Linear·SLERP**가 이해하기 쉬움. Task Arithmetic/TIES/DARE는 공통 기반 모델 기준으로 "각 모델이 어떤 방향으로 바뀌었나"를 계산해 결합 — 변화가 충돌할 수 있어 TIES·DARE가 **gradient conflict** 완화 장치를 포함

### 사용 이유

- 추가 학습 데이터·GPU·시간 없이 **여러 특화 모델의 강점 조합** (예: 한국어 모델 + 코드 모델)
- 병합 과정에 **gradient 계산 없음** — 가중치를 읽고 규칙대로 새 값 계산

> ⚠️ **항상 성능이 좋아지지 않음** — 병합 전략·비율·입력 모델 호환성·tokenizer·chat template·평가셋에 따라 결과가 크게 달라짐. **병합 실행보다 병합 후 평가가 더 중요**

### 사용 방법 — `mergekit`

병합할 모델·전략·비율·tokenizer 처리를 **YAML**에 작성 → `mergekit-yaml` 실행

| 핵심 흐름 | |
|---|---|
| 1 | 병합 목적 정하기 |
| 2 | 모델 선택 (**동일 기반 모델 권장**) |
| 3 | **tokenizer·chat template 호환 확인** |
| 4 | 병합 전략 선택 |
| 5 | YAML 작성 |
| 6 | `mergekit-yaml` 실행 |
| 7 | **평가** (병합 전후 같은 평가셋 비교) |
| 8 | 비율·전략 재조정 |

> **가장 중요한 3단계**: 모델 선택 · tokenizer/chat template 확인 · 평가. 아키텍처나 tokenizer가 안 맞으면 **전략을 바꿔도 결과가 안정적이지 않음**

> **tokenizer가 다르면?** 병합 자체는 실행되지만, 같은 문장이 다른 토큰 시퀀스로 변환되어 학습 시점과 추론 시점의 입력 구조가 어긋남. Instruct 모델은 special token·chat template이 안 맞으면 **응답 형식이 깨짐**

### 코드

```yaml
# slerp_config.yaml
slices:
  - sources:
      - model: Qwen/Qwen2.5-1.5B-Instruct          # 일반 대화
        layer_range: [0, 28]
      - model: Qwen/Qwen2.5-Coder-1.5B-Instruct    # 코드 특화
        layer_range: [0, 28]
merge_method: slerp
base_model: Qwen/Qwen2.5-1.5B-Instruct
parameters:
  t: 0.5                # 보간 비율 (0.5 = 균등)
dtype: bfloat16
tokenizer:
  source: union         # 두 tokenizer 토큰 합쳐 특수 토큰 보존
chat_template: auto
```

```bash
mergekit-yaml slerp_config.yaml ./merged-qwen-coder \
    --copy-tokenizer \      # 결과 폴더에 tokenizer 복사 (바로 추론 가능)
    --out-shard-size 1B \   # 결과를 shard로 분할 저장
    --lazy-unpickle \       # 지연 로드로 RAM 절약
    --cuda
```

```python
# 병합 결과는 일반 HF 모델처럼 로드
tokenizer = AutoTokenizer.from_pretrained("./merged-qwen-coder")
model = AutoModelForCausalLM.from_pretrained("./merged-qwen-coder", dtype="auto", device_map="auto")
```

> ⚠️ **실습 결과 주의**: 일반 Instruct + Coder를 SLERP로 섞으면 토큰 선택 방향이 충돌해 **응답이 엉뚱하게** 나올 수 있음 (한국어·중국어·영어 토큰이 어색하게 섞임). **병합 성공 ≠ 능력 결합** — 비율 `t` 조정, tokenizer·chat template 재확인, 병합 전후 평가 필수

### Model Merging vs Adapter Merge

둘 다 **추가 학습 없이 가중치 연산만으로 단일 체크포인트**를 만들지만, **합치는 대상이 다름**

| 항목 | **Model Merging** | **Adapter Merge** |
|---|---|---|
| 입력 | **여러 모델 체크포인트** | 1개 기반 모델 + **1개 어댑터** |
| 결합 대상 | 모델 가중치끼리 | 어댑터 변화량 → 원본 가중치 |
| 결합 방식 | 평균·보간·task vector | `W' = W + ΔW` |
| 목적 | **여러 특화 모델의 강점 결합** | **어댑터 없이 배포**하도록 구조 단순화 |
| 상황 | 한국어 모델 + 코드 모델 섞기 | LoRA/QLoRA 어댑터를 본체에 흡수 |

> **Adapter Merge** = 한 모델에 붙은 플러그인을 본체에 합침 / **Model Merging** = 서로 다른 완성 모델들을 가중치 수준에서 섞어 새 모델 생성
