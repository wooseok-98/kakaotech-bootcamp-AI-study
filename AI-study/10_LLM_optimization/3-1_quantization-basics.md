# 3-1. Quantization 기법 ① (Quantization · Affine · Weight Quantization)

높은 비트(FP32·FP16)로 저장된 모델 파라미터를 **정해진 규칙에 따라 낮은 비트(INT8·INT4)** 표현으로 변환하는 경량화 기법

> 단순 데이터 타입 강제 변환이 아니라, 값의 분포를 확인해 **메모리 절감 ↔ 정확도 손실의 균형**을 맞추는 최적화 과정

---

## Quantization

가중치·활성화 값·KV 캐시 같은 숫자 값을 낮은 비트 정수 칸에 맞춰 저장 (INT8=256칸, INT4=16칸) — 칸이 줄수록 메모리 ↓, 오차 ↑

### Quantization의 갈래

하나의 고정 공식이 아니라 **무엇을·언제·어떻게** 줄이느냐로 여러 축으로 나뉨

| 분류 축 | 대표 선택지 | 설명 |
|---|---|---|
| **매핑 방식** | Symmetric(대칭) / Affine(비대칭) | 정수 칸에 어떤 기준으로 맞출지 — 0 중심 대칭 vs 영점 이동 |
| **변환 대상** | Weight / Activation / KV Cache | 어떤 값을 줄일지 |
| **가중치 한정** | Weight-only | 활성화는 두고 가중치만 — 단순, LLM 추론에 자주 |
| **정밀도 표기** | W8A8 / W4A16 등 | 가중치·활성화 각 비트 수 (W8A8 = 둘 다 8비트) |
| **적용 시점** | PTQ(학습 후) / QAT(학습 중) | PTQ 쉬움, QAT는 학습 필요하나 오차↓ |
| **측정 방식** | Static / Dynamic / Calibration / Outlier | scale 기준을 언제·무슨 데이터로 정할지 |
| **대표 알고리즘** | GPTQ / AWQ | 단순 반올림보다 오차를 줄이는 고급 알고리즘 |

> INT8·INT4에서 본 단순 scale 매핑은 "가중치를 0 중심 대칭으로 보고, 학습 후 absmax로 매핑"한 **가장 기본 형태** — 실무는 대상·단위(텐서/채널/그룹)·이상치 처리까지 세분화

### 사용 이유·방법

- **이유**: 같은 하드웨어에서 더 큰 모델을 올리거나 같은 모델을 더 적은 메모리로 — 단 **품질을 크게 잃지 않는 선에서**
- **혼합 정밀도 가능**: 가중치는 INT4, 활성화는 INT8, 민감한 레이어는 FP16으로 남기는 식
- **흐름**: 목적 확인 → 대상 선택 → 비트폭·매핑 방식 결정 → 도구(bitsandbytes/GPTQ/AWQ/llama.cpp) 변환 → **성능 평가** → 필요 시 재조정

---

## Affine Quantization

**Scale factor**와 **Zero-point** 두 파라미터로 부동소수점 값을 정수로 아핀 변환(`y = ax + b`)하는 방식

| 개념 | 역할 |
|---|---|
| **Scale factor** | 값을 얼마나 압축할지 — FP 값 간격을 정수 한 칸 간격에 맞춤 |
| **Zero-point** | FP의 0이 정수 범위의 **어느 정수에 대응되는지** (영점 이동) |

### 공식

$$q = \text{clamp}(\text{round}(x / scale) + zero\_point,\ q_{min},\ q_{max})$$

$$x_{approx} = (q - zero\_point) \times scale \quad\text{(역양자화)}$$

$$scale = \frac{x_{max} - x_{min}}{q_{max} - q_{min}}, \quad zero\_point = \text{round}(q_{min} - x_{min}/scale)$$

> 역양자화 복원값은 원본과 완전히 같지 않음 (반올림) → **양자화 오차**

### 예시 (`-0.5 ~ 1.5` → INT8 `-128~127`)

```
scale = (1.5 - (-0.5)) / (127 - (-128)) = 2.0 / 255 ≈ 0.00784
zero_point = round(-128 - (-0.5)/0.00784) = round(-64.25) = -64
```

| 원래 값 x | 계산 | INT8 q |
|---|---|---|
| -0.5 | round(-0.5/0.00784 + (-64)) | -128 |
| 0.0 | round(0/0.00784 + (-64)) | **-64** |
| 1.5 | round(1.5/0.00784 + (-64)) | 127 |

> 핵심: FP `0.0`이 정수 `0`이 아니라 `-64`에 대응 — 분포가 0 중심 대칭이 아닐 때 영점을 옮겨 **정수 범위를 빈틈없이 사용**

### 사용 이유 — 비대칭 분포 대응

가중치는 0 중심 양·음 분포가 많지만, **활성화 값은 ReLU 뒤 0 이상**이거나 한쪽으로 치우침 → 대칭 양자화 쓰면 음수 정수 구간이 낭비됨. Zero-point로 실제 데이터 범위에 맞춰 오차↓

### 코드 (bitsandbytes 4bit 로드)

실무에서는 공식을 직접 호출하지 않고 도구가 Scale·Zero-point를 자동 계산

| 코드 요소 | 역할 |
|---|---|
| `BitsAndBytesConfig` | bitsandbytes 양자화 설정 객체 |
| `load_in_4bit=True` | 가중치를 4bit로 로드 (메모리↓) |
| `bnb_4bit_compute_dtype=torch.float16` | 저장은 4bit, **계산은 FP16** |

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,   # 계산용 dtype (T4는 FP16 무난)
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config, device_map="auto",
)
print(torch.cuda.memory_allocated() / 1024**3)   # 약 1.14 GB (4bit 로드)
```

> `load_in_4bit`가 "모든 연산을 4bit로" 라는 뜻은 아님 — **저장만 4bit, 연산은 compute_dtype(FP16)**

---

## Weight Quantization

학습된 **가중치**를 INT8·INT4 저비트로 바꿔 저장 크기·메모리를 줄이는 기법

> 가중치는 **학습 후 값이 고정**(정적 분포) → 활성화처럼 입력마다 분포를 다시 볼 필요 없음 → 한 번 분석해 미리 변환·저장·재사용 가능 (적용이 단순)

| 특성 | 설명 |
|---|---|
| 정적 분포 | 학습 후 고정 → 한 번 분석해 양자화 파라미터 계산 |
| 0 중심 분포 경향 | 양·음 함께 → **대칭 양자화** 자주 사용 |
| 0 근처 밀집 | 대부분 0 근처, 일부 **outlier**(유난히 큰 값)가 성능에 큰 영향 |
| 사전 변환 가능 | Scale/Zero-point 미리 계산해 파일 저장 (GPTQ·AWQ 공개 모델) |

> "정수로 저장 = 모든 연산이 정수" 아님 — 저장은 INT8/INT4, **연산은 FP16/BF16/INT32 병용**. Weight Quantization은 **저장 정밀도를 낮추는** 기법으로 이해

### Scale 단위: Per-tensor / Per-channel / Per-group

Scale을 **얼마나 크게 묶어** 계산할지의 차이 (웹 비유: 전체 공통 timeout vs 서비스별 vs 요청 그룹별)

| 방식 | 단위 | 장점 | 한계 |
|---|---|---|---|
| **Per-tensor** | 텐서 전체에 Scale 1개 | 단순, 메타데이터 최소 | 채널·구간별 분포 차이 반영 못 함 (큰 값에 기준 맞춰져 작은 값 뭉개짐) |
| **Per-channel** | 출력 채널마다 Scale | 채널별 분포 반영, 정확도↑ | 채널 수만큼 Scale 증가 |
| **Per-group** | 64·128개 등 그룹마다 Scale | INT4 저비트에서 세밀한 반영 | 메타데이터↑, 구현 복잡 |

> INT4처럼 칸이 적을수록 Per-group이 유용 — 작은 그룹 단위로 범위를 좁혀 오차↓

### 사용 이유·방법

- **이유**: 가중치가 모델 메모리의 큰 부분 → 저비트로 저장 파일·GPU 로드 메모리 직접 절감. 고정값이라 한 번 양자화 후 파일 저장·재사용 (GPTQ/AWQ 공개 모델 다운로드)
- **분포 분석 방식**:

| 방식 | 기준 |
|---|---|
| min-max | 최솟값·최댓값으로 범위 매핑 |
| absmax | 절댓값 최댓값 기준 대칭 매핑 |
| clipping | 튀는 값을 범위 안으로 잘라낸 뒤 매핑 |
| percentile | 상·하위 일부 이상치 제외 후 대부분 값 범위로 매핑 |

- **흐름**: 원본 로드 → 가중치 분포 분석 → Scale/Zero-point 계산 → 정수 변환 → (정수 가중치 + Scale/Zero-point) 저장

```python
# 원본 FP16 로드
fp16_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")
print(fp16_model.get_memory_footprint() / 1024**3)   # 약 2.88 GiB
del fp16_model; torch.cuda.empty_cache()

# 가중치만 INT8 로드 (Affine 예시의 4bit와 동일 API 계열)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
int8_model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto",
)
print(int8_model.get_memory_footprint() / 1024**3)   # 약 1.66 GiB
```

> FP16 2.88 GiB → INT8 1.66 GiB — 정확히 절반이 아니라 **절반에 가까운 수준** (일부 레이어·연산은 FP16 병용)

---

## Activation Quantization

입력에 따라 **매번 달라지는** 레이어의 중간 출력 텐서(활성화 값)를 INT8 등 저비트로 변환하는 기법

| 구분 | Weight | Activation |
|---|---|---|
| 값 | 학습 후 **고정**(정적) | 입력마다 **변함**(동적) |
| 분포 분석 | 한 번 분석해 미리 확정 | 입력마다 달라져 기준 정하기 어려움 |
| 생성 시점 | 모델 파일에 저장됨 | 추론 실행 중 생성 |

> 그래서 Activation은 Weight보다 까다로움 — "언제·무슨 데이터로 분포를 측정하느냐"가 핵심

### 사용 이유

- LLM 주요 연산 = **가중치 × 활성화 행렬곱** → 가중치만 줄이면 저장은 줄지만 연산은 여전히 FP16 경로
- 활성화까지 INT8로 줄이면 GPU/NPU/DSP의 **INT8 연산 경로 활용**(정수 연산 가속) + 추론 중 생성되는 중간 텐서 메모리도 절감

### 사용 방법 — 분포 측정 시점 2가지

| 방식 | 측정 시점 | 장점 | 단점 |
|---|---|---|---|
| 사전 측정 (Static) | 대표 데이터로 미리 측정 → Scale 고정 | 추론 시 계산 부담 적음 | 대표 데이터가 실제 입력을 못 담으면 정확도↓ |
| 실시간 측정 (Dynamic) | 추론 시점 입력별 측정 → 그때마다 Scale | 입력 변화에 유연 | 실행 중 추가 계산 오버헤드 |

> 적용 단위도 텐서 전체 / 토큰별 / 채널별로 나눌 수 있음 — 세밀할수록 정확하나 계산↑. Static/Dynamic·Calibration 상세는 [[3-2_quantization-methods]]에서 다룸

### 코드 (NumPy 개념 시연 — 대표 데이터로 Scale 고정)

```python
import numpy as np

# 1) 대표 데이터(calibration)로 활성화 분포 측정 → absmax로 scale 고정
calib = np.array([[0.0, 0.4, 1.8, 5.2], [0.1, 0.9, 2.3, 4.7], [0.0, 0.6, 3.1, 5.9]], dtype=np.float32)
qmax = 127
scale = float(np.abs(calib).max()) / qmax        # 5.9 / 127 ≈ 0.04646 (고정)

# 2) 실제 입력 활성화를 같은 scale로 양자화 → 역양자화
act = np.array([0.0, 0.7, 2.5, 5.5], dtype=np.float32)
q = np.clip(np.round(act / scale), -qmax, qmax).astype(np.int8)   # [0 15 54 118]
restored = q.astype(np.float32) * scale                            # [0. 0.697 2.509 5.482] (양자화 오차)
```

> 대표 데이터로 한 번 정한 scale을 추론에 그대로 적용 = **Static 방식** — 대표 데이터가 실제 분포를 대표해야 함

---

## KV Cache Quantization

자기회귀 추론 중 이전 토큰의 **Key/Value 벡터를 저비트(FP8·INT8)로 저장**해 KV Cache가 차지하는 GPU 메모리를 줄이는 기법

> **자기회귀 추론**: 이전 토큰들로 다음 토큰을 하나씩 생성 / **KV Cache**: 이전 토큰의 K·V를 저장해 매 토큰마다 재계산하지 않고 재사용하는 메모리 (속도↑ 대신 GPU 메모리 사용)

> KV Cache **구조를 없애는 게 아님** — 캐시는 그대로 두고, 저장되는 K/V의 **dtype만 낮춤**

### 사용 이유

- KV Cache는 가중치(고정)와 달리 **문맥 길이·요청 수에 따라 커지는 동적 메모리** — 긴 문맥·다중 요청 서빙에서 큰 비중
- FP16(2B) → FP8/INT8(1B) → KV Cache 메모리 **약 절반** → 더 긴 시퀀스 처리·더 많은 동시 요청 수용

### 사용 방법 (Transformers QuantizedCache)

`generate()` 호출 시 두 인자만 추가 (모델 가중치는 안 건드림)

| 인자 | 값 예시 |
|---|---|
| `cache_implementation` | `"quantized"` |
| `cache_config` | `{"backend": "quanto", "nbits": 4}` |

```python
outputs = model.generate(
    **inputs, max_new_tokens=100,
    cache_implementation="quantized",
    cache_config={"backend": "quanto", "nbits": 4},   # quanto/hqq 백엔드, INT4 저장
)
```

> 백엔드: `quanto`(int2/4), `hqq`(int2/4/8). **주의**: quanto INT4 경로는 `sm_80`+(A100) 요구 → T4(`sm_75`)에선 `cp.async requires sm_80` 오류 가능. 정확도는 FP8이 저하 작은 편이나 모델·문맥·엔진에 따라 다름

---

## Weight-only Quantization

**가중치만** 저비트(INT4·INT8·NF4)로 저장하고, 활성화 값은 **FP16/BF16 원래 정밀도로 유지**하는 방식

> 가중치는 학습 후 고정이라 미리 저비트 변환 가능 / 활성화는 입력마다 변해 잘못 양자화하면 정확도 손실 큼 → 그래서 가중치만 건드림

### 데이터 타입 흐름 (저장은 저비트, 연산은 FP16)

| 단계 | dtype |
|---|---|
| 저장 / GPU 로드 | INT4·INT8·NF4 (메모리↓) |
| **On-the-fly 역양자화** | 저비트 → FP16/BF16 (필요한 블록만 커널 내부에서 즉시 복원) |
| 행렬곱 / 출력 | FP16/BF16 (활성화와 함께) |

> **on-the-fly 역양자화**: 전체 가중치를 FP16으로 미리 복원하지 않고, 연산에 필요한 블록만 순간 복원 → 메모리 절감 유지하며 기존 FP16 연산 흐름과 연결. 사용자는 일반 모델처럼 API 호출

### 사용 이유

- 활성화 양자화의 복잡성·정확도 손실을 **피하면서** 가중치 메모리 절감
- **Decode 단계**(토큰 1개씩 생성, 작은 배치)는 가중치를 메모리에서 읽는 게 병목(**Memory-bound**) → 작은 INT4 가중치가 읽을 데이터량↓ → 유리 (챗봇·단건 API)
- **Prefill 단계**(프롬프트 전체 처리, **Compute-bound**)는 이점 작음 → 활성화까지 줄이는 **W8A8**이 더 나을 수 있음

```python
# bitsandbytes 4bit — 가중치만 4bit, 활성화 FP16
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model_4bit = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
print(model_4bit.get_memory_footprint() / 1024**3)   # FP16 2.88 GiB → 4bit 1.05 GiB
```

---

## W8A8 / W4A16 (비트 구성 표기)

가중치(**W**)와 활성화(**A**)를 각각 몇 비트로 표현하는지 나타내는 **표기 규약** (알고리즘 이름이 아님)

| 표기 | 가중치 / 활성화 | 대표 사용처 |
|---|---|---|
| **W8A8** | INT8 / INT8 | SmoothQuant — 메모리↓ + INT8 연산 가속 |
| **W4A16** | INT4 / FP16·BF16 | GPTQ·AWQ·NF4 — Weight-only, 메모리 절감 중심 |
| W8A16 | INT8 / FP16 | Weight-only·혼합 정밀도 |
| W4A8 | INT4 / INT8 | 메모리+가속 동시 노림 (실험적) |

> **왜 따로 표기?** 가중치는 고정·안정적이라 4bit까지 **공격적으로** 축소 가능, 활성화는 입력마다 변하고 outlier 잦아 **보수적으로** 유지 → W4A16처럼 비대칭 구성이 흔함

### 해석 용도 (실행 전 1차 판단)

모델명·모델카드·논문·추론엔진 설정에서 표기를 읽어 **메모리/속도/하드웨어 호환성**을 미리 판단

| 태그 예시 | 해석 | W/A |
|---|---|---|
| `GPTQ-Int4` | 가중치 4bit, 활성화 FP16 | W4A16 |
| `AWQ` | 중요 가중치 보호하며 4bit | 보통 W4A16 |
| `bnb-4bit` | bitsandbytes 4bit 로드 | 보통 W4A16 |
| `SmoothQuant W8A8` | 가중치·활성화 모두 8bit | W8A8 |

> **KV 표기 확장**: `W4A16KV8` = 가중치 4bit · 활성화 16bit · **KV Cache 8bit**

> 관련 알고리즘 GPTQ·AWQ 상세는 [[4-1_advanced-quantization]], PTQ/QAT/Calibration/Outlier는 [[3-2_quantization-methods]]에서 다룸
