# LLM Optimization 한줄 정리

## 개요

| 개념 | 한줄 정리 |
|------|------|
| LLM Optimization | 학습 끝난 모델을 재학습 없이 크기·속도·메모리 관점에서 가볍게 만드는 기법 |
| 핵심 3축 | 데이터타입/Quantization(크기↓) · PEFT(효율적 재학습) · 배포형식 GGUF(실행 환경 적응) |

## 데이터 타입

| 개념 | 한줄 정리 |
|------|------|
| FP32 | 32비트(부호1·지수8·가수23) 단정밀도 부동소수점, 딥러닝 기본·가장 안정 |
| FP16 | 16비트(1·5·10) 반정밀도, 메모리 절반이지만 범위 좁아 오버플로 위험 |
| BF16 | 16비트(1·8·7), 지수부 FP32와 동일해 범위 넓음 → 대규모 학습에 안정적 |
| Mixed Precision Training | 연산은 FP16/BF16, 가중치 누적은 FP32로 분업하는 학습 |
| INT8 | 8비트 정수, 실수를 256개 정수 칸에 scale로 매핑, FP32의 1/4 메모리 |
| INT4 | 4비트 정수, 16개 칸에 매핑, FP32의 1/8 메모리, 오차는 크지만 대형 모델에 유용 |

## Quantization 기법 ① (무엇을·어떻게)

| 개념 | 한줄 정리 |
|------|------|
| Quantization | 높은 비트(FP32/16) 파라미터를 규칙에 따라 낮은 비트(INT8/4)로 매핑하는 경량화 |
| Affine Quantization | Scale(간격)과 Zero-point(0 위치)로 비대칭 분포도 정수 범위에 빈틈없이 매핑 |
| Weight Quantization | 학습 후 고정된 가중치를 저비트로 저장, 미리 분석 가능해 적용이 단순 |
| Activation Quantization | 입력마다 변하는 중간 출력값을 저비트로, 기준을 언제 정할지가 관건 |
| KV Cache Quantization | 자기회귀 추론의 이전 토큰 K/V를 FP8/INT8로 저장해 KV Cache 메모리 절감 |
| Weight-only Quantization | 가중치만 줄이고 활성화는 FP16 유지(=W4A16) |
| W8A8 / W4A16 | 가중치·활성화 비트 수 표기, W8A8=정수 연산 가속, W4A16=Weight-only 메모리 절감 |
| Per-tensor/channel/group | Scale을 텐서 전체/채널/그룹 단위로 계산, 잘게 나눌수록 오차↓ |

## Quantization 기법 ② (언제·기준)

| 개념 | 한줄 정리 |
|------|------|
| PTQ (Post-Training Quantization) | 학습 끝난 모델을 재학습 없이 저비트로 변환하는 사후 양자화 |
| QAT (Quantization-Aware Training) | 학습 중 Fake Quant로 오차를 미리 겪게 해 웨이트를 양자화에 강하게 적응시키는 방식 - 그 후 확정된 웨이트는 양자화해도 성능이 덜 떨어짐 |
| Fake Quantization | QAT에서 FP32 위에 양자화↔역양자화를 흉내 내 순전파에 오차를 주입하는 노드 |
| Static Quantization | 대표 데이터로 활성화 scale을 사전 측정·고정해 추론 시 재계산 없음 |
| Calibration | 대표 입력을 통과시켜 활성화 분포를 측정해 scale·zero-point를 정하는 절차 |
| Dynamic Quantization | 추론 시점에 입력마다 활성화 scale을 즉석 계산, NLP처럼 분포 변동 클 때 유리 |
| Outlier Problem | 소수의 극단값이 scale을 키워 일반 값 표현을 뭉개는 저비트 정확도 급락 원인 |
| STE (Straight-Through Estimator) | round의 기울기 0 문제를 역전파에서 dy/dx≈1로 근사해 학습 신호를 통과시키는 방법 |

## 고급 Quantization

| 개념 | 한줄 정리 |
|------|------|
| RTN (Round-to-Nearest) | 각 가중치를 독립적으로 반올림하는 가장 단순한 양자화, INT4에서 오차 누적으로 품질 급락 |
| GPTQ | 한 가중치 양자화로 생긴 오차를 아직 양자화 안 한 가중치에 분배하는 PTQ (사후 보상) |
| Hessian | 가중치가 바뀔 때 출력이 얼마나 흔들리는지 알려주는 민감도 지도, GPTQ가 오차 분배에 사용 |
| AWQ | 활성화 통계로 중요 채널을 찾아 스케일링으로 미리 보호하는 PTQ (사전 보호) |
| Salient Weight | 값이 큰 가중치가 아니라, 활성화가 큰 채널에 연결된 출력 영향이 큰 가중치 |
| Quantization Evaluation | 양자화 전후 품질(PPL·벤치마크)과 서빙 효율(TTFT·VRAM)을 같은 조건에서 비교해 배포 판단 |
| Perplexity (PPL) | 다음 토큰 예측 품질 지표, 낮을수록 좋고 양자화 평가의 1차 지표로 사용 |

## Formats & Tools

| 개념 | 한줄 정리 |
|------|------|
| GGUF | 가중치·아키텍처·토크나이저·양자화 설정을 하나의 바이너리에 담은 자기완결적 파일 형식 |
| k-quant (Q4_K_M 등) | GGUF 안의 양자화 저장 타입으로, 텐서마다 다른 비트를 배분해 크기·품질 균형을 맞춤 |
| Llama.cpp | GGUF 모델을 CPU·Metal·CUDA 등 다양한 환경에서 실행하는 C/C++ 기반 경량 추론 엔진 |
| `-ngl` (n-gpu-layers) | GPU에 오프로드할 레이어 수, 레이어는 순차 처리되고 계산 위치만 달라짐 |

## PEFT 기법

| 개념 | 한줄 정리 |
|------|------|
| PEFT | 사전학습 모델 대부분을 고정하고 소수의 추가 파라미터만 학습하는 파인튜닝 기법들의 총칭 |
| LoRA | 원본 가중치는 고정하고 변화량 ΔW를 두 저랭크 행렬 곱으로 근사해 1% 미만만 학습 |
| rank (r) | LoRA 어댑터의 표현력을 정하는 저랭크 차원, 클수록 표현력↑ 파라미터·메모리↑ |
| lora_alpha | 어댑터 보정값의 반영 강도를 조절하는 스케일링, 실제 강도는 lora_alpha/rank 비율 |
| QLoRA | 기반 모델을 4-bit NF4로 양자화해 고정하고 16-bit LoRA 어댑터만 학습하는 기법 |
| NF4 (NormalFloat 4-bit) | 가중치가 0 근처에 몰린다는 특성을 반영해 구간을 비균등 배치한 4비트 타입 |
| Double Quantization | 양자화 상수(scale)까지 한 번 더 압축해 부가 메모리를 줄이는 기법 |
| Paged Optimizer | 옵티마이저 상태를 GPU 부족 시 CPU RAM으로 옮겼다 가져와 OOM을 방지 |
| Adapter Merge | 어댑터 변화량을 원본 가중치에 합산(W'=W+ΔW)해 단일 모델 체크포인트로 굳히는 과정 |
| Model Merging | 여러 모델 체크포인트의 가중치를 추가 학습 없이 결합해 새 단일 모델을 만드는 기법 |
| SLERP / TIES / DARE | 구면 보간 / 충돌 방향 정리 / 변화량 드롭 후 재조정 — 대표적인 모델 병합 전략 |
