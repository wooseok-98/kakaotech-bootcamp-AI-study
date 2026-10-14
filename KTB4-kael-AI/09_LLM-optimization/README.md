# 09. LLM Optimization

Qwen2.5-1.5B-Instruct 를 대상으로 **PEFT → PTQ → GGUF 배포** 파이프라인을 실습한 과제

---

## 노트북

| 파일 | 주제 | 런타임 |
| --- | --- | --- |
| [01_lora_qlora.ipynb](./01_lora_qlora.ipynb) | LoRA · QLoRA 파인튜닝, 자원 사용량 비교, 어댑터 병합 | GPU (T4+) |
| [02_ptq.ipynb](./02_ptq.ipynb) | bnb-INT8 · NF4 · GPTQ-INT4 양자화, 성능·메모리 비교 | GPU (T4+) |
| [03_gguf_llamacpp.ipynb](./03_gguf_llamacpp.ipynb) | GGUF 변환, K-quant 양자화, Llama.cpp 추론·벤치마크 | CPU 가능 |

---

## 파이프라인

```
Qwen2.5-1.5B-Instruct (fp16)
  │
  ├─[01] LoRA  학습 ─→ adapter A ┐
  └─[01] QLoRA 학습 ─→ adapter B ┘ 자원·성능 비교
                          │
                   merge (fp16 베이스에 병합)
                          │
                   merged fp16 모델  ──→ HF Hub (private)
                     ├─[02] GPTQ / bitsandbytes  → GPU 서빙용
                     └─[03] GGUF Q4_K_M          → CPU · 로컬 실행용
```

**핵심 설계** — `02` 와 `03` 은 둘 다 merged fp16 을 입력으로 받는 **독립 노트북**
- 세션이 끊겨도 앞 단계를 다시 돌릴 필요 없음
- 같은 원본에서 갈라진 두 양자화 경로를 나란히 비교하는 구도가 됨

> llama.cpp 는 GPTQ · bitsandbytes 가중치를 읽지 못함
> GGUF 의 K-quant 는 그 자체가 독립적인 PTQ 경로

---

## 실행 전 설정

| 항목 | 내용 |
| --- | --- |
| `HF_USER` | 각 노트북 상단 TODO 에 본인 HuggingFace 아이디 입력 |
| `HF_TOKEN` | Colab 좌측 🔑 시크릿에 **write 권한** 토큰 등록 |
| Drive | `MyDrive/ktb/09_llm_opt` 에 지표·어댑터·GGUF 저장 |

---

## 산출물

| 파일 | 생성 노트북 | 내용 |
| --- | --- | --- |
| `metrics_train.csv` | 01 | LoRA vs QLoRA (VRAM · 학습시간 · loss) |
| `metrics_ptq.csv` | 02 | fp16 vs INT8 vs NF4 vs GPTQ (용량 · VRAM · tok/s · PPL) |
| `metrics_gguf.csv` | 03 | GGUF 양자화 타입별 용량 |
| `eval_texts.json` | 01 | 02 의 PPL 평가 · GPTQ 캘리브레이션 공용 데이터 |
| `qwen-Q4_K_M.gguf` | 03 | 로컬(Apple Silicon) 실행용 최종 모델 |

---

## 짚어볼 지점

**QLoRA 어댑터 병합**
4bit 베이스 위에서 학습했더라도 병합은 **fp16 베이스로 재로딩 후** 수행해야 함
양자화된 가중치에 어댑터를 더하면 값이 깨짐

**QLoRA 가 더 느릴 수 있음**
매 연산마다 4bit → 16bit 역양자화 비용이 붙음 — 메모리를 시간과 맞바꾼 구조

**INT8 이 fp16 보다 느릴 수 있음**
outlier 를 fp16 으로 분리 처리하는 혼합 연산 오버헤드

**용량이 정확히 1/4 로 안 줄어듦**
임베딩 · LayerNorm 등은 양자화 대상에서 제외

**PPL 만 보지 말 것**
수치가 비슷해도 생성 문장이 무너질 수 있어 정성 비교 병행

---

## 관련 학습 노트

[AI-study/10_LLM_optimization](../../AI-study/10_LLM_optimization) — 데이터 타입 · Quantization · PEFT · GGUF 이론 정리
