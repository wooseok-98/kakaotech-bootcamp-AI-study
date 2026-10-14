# 3-2. Quantization 기법 ② (PTQ · Static/Dynamic · Calibration)

양자화를 **언제·무슨 데이터로** 적용하느냐를 다루는 축 — 학습 후(PTQ) vs 학습 중(QAT), 활성화 기준을 사전 고정(Static) vs 실시간 계산(Dynamic)

> 3-1이 "무엇을(weight/activation/KV)" 줄이나였다면, 3-2는 "**어떻게·언제 기준을 정하나**"

---

## PTQ (Post-Training Quantization)

**학습이 끝난 모델을 재학습 없이** 가중치·활성화 값을 INT8·INT4 저비트로 변환하는 사후 양자화 기법

> 특정 알고리즘이 아니라 **적용 시점 기준 분류** — 학습 후 적용하면 GPTQ·AWQ·SmoothQuant·bitsandbytes 4bit 모두 넓은 의미의 PTQ 계열

### 동작 과정

| 순서 | 단계 |
|---|---|
| 1 | 원본 모델 로드 (FP32/FP16/BF16) |
| 2 | Calibration 데이터 준비 (방식에 따라 생략) |
| 3 | 대표 데이터 통과시켜 분포 측정 |
| 4 | Scale·Zero-point 등 양자화 파라미터 결정 |
| 5 | 저비트 변환·저장 |

### 사용 이유

- **재학습 불필요** → 대규모 학습 데이터·장시간·다중 GPU 인프라 없이 빠르게 적용. 공개 모델에 양자화만 얹어 배포 검토
- 단, 학습 중 오차 보정을 안 하므로 **비트를 극단적으로 낮추면 정확도 손실↑**
  - **INT8**: PTQ만으로 FP16과 유사한 경우 많음
  - **INT4 이하**: GPTQ·AWQ 같은 오차 보정 고급 PTQ 병행 권장
- "빠르게 배포"는 재학습 대비 부담이 작다는 뜻 — 실제론 반드시 **메모리·속도·정확도 측정** 후 결정

> 실무 최소 흐름: bitsandbytes 4bit 로드 (NF4) — calibration 없이 가중치만 4bit, Qwen2.5-1.5B FP16 2.88 → 4bit 1.05 GiB (= [[3-1_quantization-basics]] Weight-only). 정량 평가는 [[4-1_advanced-quantization]] Evaluation

---

## Static Quantization vs Dynamic Quantization

**활성화 값의 양자화 기준(scale)을 언제 정하느냐**로 갈림 (가중치는 둘 다 사전 양자화 — 고정값이라)

> 가중치는 학습 후 고정이라 미리 분석 가능 / 활성화는 입력마다 새로 생기는 중간값 → "미리 정할까 vs 그때그때 정할까" 선택 발생

| 구분 | Static | Dynamic |
|---|---|---|
| 활성화 기준 | **Calibration으로 사전 측정 → 고정** | **추론 시점 입력마다 즉석 계산** |
| Calibration | 필요 | 불필요 |
| 추론 오버헤드 | 없음 (저장된 scale 재사용) | 있음 (매 입력 분포 측정+계산) |
| 지연 시간 | 일정 | 입력별 계산 추가 |
| 유리한 환경 | 입력 분포 일정 — 이미지 분류, 온디바이스, 모바일 NPU/CPU 정수 백엔드 | 입력 분포 다양 — **NLP**(문장 길이·언어·도메인 변동 큼) |

```
Static:  [준비 1회] calib 데이터 → 분포 측정 → scale 저장 ┈┈→ [추론 매번] 저장된 scale로 바로 정수 연산
Dynamic: [추론 매번] 입력 → 활성화 생성 → 분포 즉석 측정 → scale 계산 → 정수 연산
```

> NLP에서 Dynamic 선호: 짧은 질문·긴 요약·다른 언어처럼 활성화 분포가 크게 달라져 고정 기준(Static)이 안 맞을 위험 → 입력마다 맞추는 게 유연

### 코드 (PyTorch CPU Static: prepare → calibrate → convert)

> CPU 양자화 백엔드 `fbgemm`(x86) 사용 시연용 — 실제 GPU LLM 양자화는 Weight-only/GPTQ/AWQ. 활성화 진입/종료점을 `QuantStub`/`DeQuantStub`로 명시

```python
import torch, torch.nn as nn
import torch.ao.quantization as tq

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = tq.QuantStub()      # 양자화 영역 진입점 (FP → INT)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dequant = tq.DeQuantStub()  # 종료점 (INT → FP)
    def forward(self, x):
        x = self.quant(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.dequant(x)

model = SimpleModel().eval()
model.qconfig = tq.get_default_qconfig("fbgemm")
prepared = tq.prepare(model)            # observer 삽입

with torch.no_grad():                   # calibration — 대표 데이터로 활성화 분포 측정
    for _ in range(100):
        prepared(torch.randn(1, 768))

quantized = tq.convert(prepared)        # 측정 분포로 scale/zero-point 계산 후 정수 모델 변환
```

> `prepare`(observer 삽입) → `calibration`(분포 측정) → `convert`(scale 확정) — 이후 추론은 고정 scale로 즉시 정수 연산

---

## Calibration

대표 입력 데이터를 모델에 통과시켜 **활성화 값의 통계(범위·분포)를 측정**하고, 이를 바탕으로 양자화 파라미터(scale·zero-point)를 결정하는 절차

> 학습이 아니라 **통계 측정** — 순전파만 수행, 정답 라벨 불필요

### 왜 활성화만 Calibration이 필요한가

| | 가중치 | 활성화 |
|---|---|---|
| 성격 | 학습 후 **고정**(정적) | 입력마다 생성(**동적**) |
| 분포 파악 | 모델 파일에서 바로 계산 | 실제 입력 통과시켜 관찰 필요 |

### Calibration 데이터 조건

| 조건 | 설명 |
|---|---|
| **소량** | 학습 아님 → 수십~수백 개. GPTQ·AWQ는 보통 128~512개 |
| **대표성** | 실제 추론 입력 분포 반영 (안 맞으면 정확도↓) |
| 레이블 불필요 | 입력만 있으면 됨 |
| **도메인 일치** | 코드 모델=코드, 한국어 서비스=한국어 |

> 안경 비유: 남의 시력에 맞춘 안경을 내가 쓰면 흐림 — calibration 데이터가 실제 입력과 달라지면 미리 정한 scale이 안 맞아 오차↑

### 왜 중요한가 (범위 배분)

```
실제 활성화 범위 -3~5 인데 임의로 -10~10 으로 scale 잡으면:
-10    -3            5    10
 |------|===실제값===|------|
   낭비     집중        낭비        → 정수 칸이 빈 구간에 낭비, 실제 구간 정밀도↓

Calibration으로 -3~5 측정하면 INT8 256칸을 그 구간에 집중 → 같은 비트로 더 촘촘 → 오차↓
```

### Observer (PyTorch 통계 수집기)

| Observer | 방식 | 용도 |
|---|---|---|
| `MinMaxObserver` | 전체 min/max 기록 | 가장 단순 |
| `MovingAverageMinMaxObserver` | min/max 이동평균 | 분포가 흔들릴 때 안정적 |
| `HistogramObserver` | 분포를 히스토그램으로 | **outlier** 있을 때 범위 정교하게 |
| `PerChannelMinMaxObserver` | 채널별 min/max | 채널별 범위 차 큰 가중치 |

```python
observer = tq.MinMaxObserver(dtype=torch.qint8)
observer(x)                                   # 대표 입력 관찰 (min/max 기록)
scale, zero_point = observer.calculate_qparams()   # 기록된 통계로 파라미터 계산
```

> GPTQ·AWQ 같은 LLM 도구는 내부에서 calibration을 처리해 observer가 코드에 안 드러날 수 있음

```python
observer = tq.MinMaxObserver(dtype=torch.qint8)
with torch.no_grad():
    for x in calibration_data:      # 대표 입력 통과 → min/max 기록
        observer(x)
scale, zero_point = observer.calculate_qparams()   # 통계로 파라미터 산출
```

---

## QAT (Quantization-Aware Training)

추론 시 적용될 **양자화 오차를 학습 과정에서 미리 시뮬레이션**해, 저비트 추론에서도 정확도를 유지하도록 모델을 적응시키는 학습 기법

> PTQ(학습 후 사후 적용)와 달리, 학습 중에 모델이 양자화 오차를 **미리 경험**하게 함

### Fake Quantization (가짜 양자화)

학습 그래프에 삽입하는 노드 — 실제 INT 연산을 하는 게 아니라 **FP32 위에서 양자화↔역양자화를 흉내** 내 오차를 반영

```
1.27 (FP32) → 정수 칸에 반올림(round) → 범위 밖 자르기(clamp) → 1.25 (오차 포함 FP32) → 순전파
```

$$x_{out} = (\text{clamp}(\text{round}(x/scale + zp),\ q_{min},\ q_{max}) - zp) \times scale$$

> 정수로 바꿨다 다시 FP32로 되돌렸을 때 생기는 오차를 흉내 — 모델은 "나중에 저비트로 바뀌면 값이 이렇게 틀어지겠구나"를 학습

### 순전파 / 역전파 역할 분리

| 단계 | 동작 |
|---|---|
| 순전파 | Fake Quant로 **오차 포함된 값**으로 연산 (저비트 손실 미리 경험) |
| 역전파 | `round`는 미분 안 되므로 **STE**로 근사 |
| 가중치 업데이트 | **FP32 마스터 가중치** 업데이트 (실제 INT 학습 아님) |

### STE (Straight-Through Estimator)

순전파는 `round` 그대로 적용, 역전파는 **`round`가 없던 것처럼 `dy/dx ≈ 1`로 기울기 통과**

> `round`는 계단 함수라 기울기가 대부분 0 → 그대로 쓰면 학습 신호가 앞 레이어로 안 감 → STE로 우회

### 사용 이유·방법

- **언제**: PTQ만으로 목표 정확도 미달 시. INT8은 PTQ로 충분, INT4도 GPTQ·AWQ로 대개 해결 → **INT3·INT2 극저비트**에서 QAT 가치 커짐
- **흐름**: `qconfig`(양자화 설정) → `prepare_qat`(Fake Quant·observer 삽입) → **낮은 lr로 추가 학습** → `convert`(실제 INT 모델 변환) → 재평가
- 학습(FP32+Fake Quant)과 배포(실제 INT)가 **분리**됨. PTQ보다 비용 큼(데이터·GPU·시간) → **PTQ 먼저, 부족하면 QAT**

> QAT는 기본값이 아니라 **저비트 정확도 보강용 추가 학습 수단**

---

## Outlier Problem

텐서 속 **소수의 극단값(outlier)이 양자화 scale을 끌어올려** 일반 값의 표현 해상도를 잠식 → 저비트 정확도 급락

> **표현 해상도**: 같은 비트로 표현하는 격자가 얼마나 촘촘한지 (scale 작을수록 촘촘)

### 원리 — 이상치 하나가 scale을 망침

```
대부분 -1~1 인데 값 하나가 30 → 범위를 30까지 넓힘 → 정수 칸 간격 확 커짐
→ -0.8, -0.2, 0.1, 0.7 이 전부 같은 칸으로 뭉개짐 (구분 불가)
```

| 상황 | 표현 범위 | 칸 간격 | 일반 값 |
|---|---|---|---|
| 이상치 없음 | -1~1 | 좁음 | 촘촘히 구분 |
| 이상치 `30` 1개 | -1~30 | 매우 넓음 | 대부분 같은 칸에 뭉개짐 |

> 색상 팔레트 비유: 대부분 어두운색인데 밝기 255 하나 때문에 팔레트를 0~255로 펴면, 자주 쓰는 어두운 구간이 뭉개짐. **문제는 이상치 크기가 아니라 "전체 scale을 키운다"는 점**

### LLM에서 나타나는 위치

특정 **채널·토큰·레이어에 집중** (무작위 아님) — activation 특정 채널, LayerNorm 직후, KV Cache, 가중치 일부 row/column

> 텐서 전체 평균만 보면 일반 값에 희석돼 놓침 → **채널별 최댓값·토큰별 분포**를 봐야 함. INT8은 버티지만 **INT4 이하·activation 양자화(W8A8)** 에서 치명적

### 해결 — 제거❌, 격리·분리·평활

이상치도 중요 정보를 담을 수 있어 **제거하면 정확도↓** → 영향 범위를 줄이는 방향

| 해결 | 아이디어 | 적합 |
|---|---|---|
| **Per-channel** | 채널마다 scale → 이상치 영향을 그 채널에 가둠 | 기본 INT8·INT4 |
| **Per-group** | 채널을 더 작은 그룹으로 나눠 scale | INT4 이하 |
| **Outlier 분리** | 이상치 채널만 FP16 고정밀로 따로 (`LLM.int8()`) | 대규모 INT8 |
| **Activation Smoothing** | 활성화 이상치를 **가중치 쪽으로 이동**해 분포 평탄화 (`SmoothQuant`) | W8A8 등 활성화 양자화 |

> 여러 양자화 기법(Per-group, GPTQ, AWQ, SmoothQuant)은 결국 **outlier로 인한 표현 해상도 손실을 줄이는 서로 다른 전략** — 실무는 Per-channel/group → GPTQ·AWQ → SmoothQuant → LLM.int8() 순으로 검토. GPTQ·AWQ 상세는 [[4-1_advanced-quantization]]