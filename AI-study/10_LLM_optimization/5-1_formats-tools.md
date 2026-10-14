# 5-1. Optimization Formats & Tools (GGUF · Llama.cpp)

최적화된 모델을 **로컬·경량 환경에서 바로 실행**할 수 있도록 만든 파일 형식과 런타임

> 3축 중 **배포 형식** — GPU 서버가 아닌 CPU·노트북·Apple Silicon에서도 LLM을 돌리기 위한 갈래

---

## GGUF (GGML Universal File)

llama.cpp의 표준 모델 형식 — **가중치 텐서 · 모델 아키텍처 · 토크나이저 · 양자화 설정**을 하나의 바이너리 파일에 담은 **자기완결적** 파일 형식

> **GGML** = Georgi **G**erganov(GG) + **M**achine **L**earning(ML) — C/C++ 기반 ML 라이브러리이자 llama.cpp 생태계의 기반

### HF 형식 vs GGUF

| | Hugging Face | GGUF |
|---|---|---|
| 가중치 | `.safetensors` | **`.gguf` 하나에 전부** |
| 모델 구조 | `config.json` | (메타데이터에 포함) |
| 토크나이저 | `tokenizer.json` 등 | (메타데이터에 포함) |

> 런타임(llama.cpp·Ollama·LM Studio)이 **`.gguf` 파일 하나만 보고** 모델 구조·토크나이저·양자화 타입을 모두 파악 — 여러 설정 파일을 맞출 필요 없음

### 파일 구조

로더가 읽는 순서: **헤더 → 메타데이터 → 텐서 정보 → 텐서 데이터**

| 영역 | 내용 | 역할 |
|---|---|---|
| **Header** | 매직 넘버(`"GGUF"`), 버전, 텐서 개수, 메타데이터 개수 | 파일 형식 확인·파싱 시작 정보 |
| **Metadata (K-V)** | 아키텍처, 레이어 수, 히든 사이즈, 헤드 수, 컨텍스트 길이, 토크나이저, 양자화 타입 | 모델 구조·실행 설정 |
| **Tensor Info** | 텐서 이름·차원·데이터 타입·오프셋 | 가중치를 찾고 해석하는 **목차** |
| **Tensor Data** | 실제 가중치 바이트 (F16/Q8_0/Q4_K_M 등으로 인코딩) | 추론에 쓰이는 파라미터 본체 |

> Tensor Info가 없으면 바이트를 어떻게 잘라 읽을지 알 수 없음 — 같은 바이트도 차원·타입을 모르면 복원 불가

### 양자화 타입

GGUF는 **파일 형식**이고, 양자화 타입은 그 안의 텐서가 어떤 방식으로 압축됐는지 나타내는 정보 (파일명에 표기)

| 타입 | 비트 경향 | 품질 | 비고 |
|---|---|---|---|
| Q2_K | ~2~3 | 낮음 | 가장 작지만 손실 큼 |
| Q3_K_M | ~3~4 | 보통 | 매우 가벼운 실행 |
| **Q4_K_M** | ~4~5 | 양호 | **크기·품질 균형, 입문 기본값** |
| Q5_K_M | ~5~6 | 우수 | Q4보다 크지만 안정적 |
| Q6_K | ~6 | 매우 우수 | 메모리 여유 시 |
| Q8_0 | ~8 | 최상 | 원본에 가까움 (≈ INT8) |
| F16 | 16 | 원본 수준 | 양자화 안 함 (≈ FP16) |

> `llama-2-7b.Q4_K_M.gguf` = Llama 2 7B를 GGUF로 저장 + 가중치를 Q4_K_M로 양자화

### k-quant 계열 (`_K`)

`Q4_K_M`처럼 `_K`가 붙은 타입 — GPTQ·AWQ 같은 **알고리즘이 아니라**, GGUF 안에 텐서를 저장하는 **양자화 타입**

> **핵심 특징**: 텐서마다 **다른 비트 수준**을 사용 가능 — 중요한 텐서는 비트를 더 남기고 덜 중요한 텐서는 더 줄여 크기·품질 균형. 블록 단위로 묶어 `scale` 같은 보조 정보를 함께 저장해 오차 감소

| 표기 | 의미 |
|---|---|
| `Q4` | 4비트 중심 |
| `_K` | k-quant 계열 |
| `_S` / `_M` / `_L` | Small(작게) / Medium(균형) / Large(품질) 변형 |

> `Q4_K_M` = 4비트 k-quant 중 **크기·품질 균형(Medium)** 변형. S/M/L 세부 규칙은 도구 버전·모델 구조에 따라 달라져 **고정 공식으로 외우지 말 것**

### k-quant vs GPTQ vs AWQ

| 항목 | k-quant (GGUF) | GPTQ | AWQ |
|---|---|---|---|
| 성격 | **저장 타입** | 양자화 **알고리즘** | 양자화 **알고리즘** |
| 전략 | 텐서별 비트 배분 차등 | Hessian 오차 보상 | 활성화 통계 채널 스케일링 |
| Calibration | 기본 불필요 (`imatrix` 선택적) | 필요 | 필요 |
| 실행 환경 | CPU·Metal·CUDA·Vulkan 등 | 주로 GPU | 주로 GPU |
| 런타임 | llama.cpp, Ollama, LM Studio | Transformers, vLLM | vLLM, Transformers |

> **imatrix** (importance matrix): calibration 데이터로 어떤 가중치가 중요한지 측정한 통계 — Q4 이하·IQ 계열 저비트에서 품질 보존에 도움

### 사용 이유

- **파일 하나**로 실행 정보 전부 → 다른 환경으로 옮기기 쉬움
- **넓은 실행 환경**: CPU, Apple Silicon, CUDA, Metal, Vulkan — 고가 GPU 서버 없이 로컬 LLM 실행
- **양자화 타입 선택 자유**: 메모리 부족하면 가벼운 타입, 품질 우선이면 높은 비트

> 같은 GGUF라도 실행 가능 여부는 RAM·컨텍스트 길이·KV 캐시 크기·OS 메모리에 좌우 — 처음엔 **한 단계 가벼운 타입부터** 시도

### GGUF 준비 3가지 경로

| 경로 | 방식 |
|---|---|
| **A** | Hugging Face `-GGUF` 리포에서 `.gguf` 다운로드 (가장 빠름) |
| **B** | HF 모델 → `convert_hf_to_gguf.py`(F16) → `llama-quantize`(Q4_K_M 등) |
| **C** | Ollama로 import (Ollama 생태계 안에서 관리) |

### 코드 (경로 B — 직접 변환)

```bash
git clone https://github.com/ggml-org/llama.cpp
pip install -q gguf transformers sentencepiece protobuf   # 변환 스크립트는 순수 파이썬
cmake -S llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build --target llama-quantize -j   # 양자화 도구만 빌드 (C++)
```

```bash
# ① HF 모델 폴더 → F16 GGUF (config.json·토크나이저 포함 폴더 전체가 입력)
python llama.cpp/convert_hf_to_gguf.py Qwen2.5-0.5B-Instruct \
    --outtype f16 --outfile qwen2.5-0.5b-instruct-f16.gguf

# ② F16 GGUF → Q4_K_M GGUF (입력, 출력, 양자화 타입)
llama.cpp/build/bin/llama-quantize \
    qwen2.5-0.5b-instruct-f16.gguf \
    qwen2.5-0.5b-instruct-q4_k_m.gguf \
    Q4_K_M          # Q5_K_M, Q8_0 등으로 교체 가능
```

| 파일 | 크기 |
|---|---|
| `...-f16.gguf` | 948.1 MB |
| `...-q4_k_m.gguf` | **379.4 MB** |

> **정확히 1/4로 안 줄어듦** — 헤더·메타데이터·토크나이저·일부 고정밀 텐서가 함께 들어있기 때문. `falling back to q5_0` 경고는 텐서 차원이 해당 타입 조건(256의 배수)에 안 맞아 다른 타입으로 자동 대체된 것

### 코드 (경로 A — 사전 변환본 다운로드)

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",       # 리포 이름 끝에 -GGUF
    filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",    # 원하는 양자화 타입 파일
)
# HF 캐시 폴더(/root/.cache/huggingface/hub/...)에 저장됨
```

> 한 리포에 `Q4_K_M`·`Q5_K_M`·`Q8_0` 등 여러 양자화 파일이 함께 올라와 있는 경우가 많음

---

## Llama.cpp

Meta의 Llama 모델 추론에서 출발해, 다양한 LLM 아키텍처를 **GGUF 파일로 실행**할 수 있게 해주는 **C/C++ 기반 경량 추론 엔진**

> **추론 엔진**: 학습이 아니라 **학습이 끝난 모델을 실행**해 결과를 생성하는 프로그램 — 모델 파일이 데이터라면 추론 엔진은 그걸 읽어 요청을 처리하는 서버 런타임

### 핵심 특징

| 특징 | 설명 |
|---|---|
| 추론 전용 | 학습 X, 준비된 모델 실행에 집중 |
| **C/C++ 기반** | Python·PyTorch 같은 무거운 런타임 불필요 |
| GGUF 중심 | GGUF를 표준 입력 형식으로 사용 |
| 다양한 백엔드 | CPU, CUDA, Metal, HIP, Vulkan |
| 다양한 모델 | Llama뿐 아니라 **Qwen·Mistral·Gemma·Phi** 등 |

> 이름에 `Llama`가 있지만 Llama 전용이 아님. Ollama·LM Studio도 llama.cpp 기반/호환 런타임을 활용

### 사용 이유

- **고가 GPU 서버 없이** 로컬 PC·Mac에서 LLM 테스트·검증·배포
- **프라이버시**: 로컬 실행이라 입력이 외부 API로 전송되지 않음 (의료·법률·금융·사내 문서)

| 환경 | 일반적 가능 범위 |
|---|---|
| 일반 노트북 (CPU, 8GB) | 7B `Q4_K_M` 시도 가능하나 여유 거의 없음 |
| MacBook Air M2 (16GB) | 7B~13B `Q4_K_M` + **Metal 가속** |
| MacBook Pro M3 Max (96GB) | 70B `Q4_K_M`까지 시도 가능 |
| 데스크톱 (32GB + RTX 4090) | 13B 고비트 / 70B 저비트 (CPU+GPU 분할) |

> **Metal 가속**: Apple GPU 연산 기술. Apple Silicon은 CPU·GPU가 통합 메모리를 공유해 연산을 GPU로 넘길 수 있음

### 사용 방법

`CMake 빌드 → GGUF 확보 → llama-cli/llama-completion(CLI) 또는 llama-server(API)`

```bash
apt-get install -y libcurl4-openssl-dev        # -hf 옵션(HF 직접 다운로드)에 필요
git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75   # T4=sm_75. macOS는 Metal 기본 활성
cmake --build build --config Release --target llama-completion -j 2
cmake --build build --config Release --target llama-server -j 2
```

```bash
# 단발 추론
build/bin/llama-completion \
    -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -p "어댑터즈는" -n 128 -ngl 99
```

| 옵션 | 의미 |
|---|---|
| `-m` | GGUF 모델 파일 경로 |
| `-p` | 입력 프롬프트 |
| `-n` | 최대 생성 토큰 수 |
| `-ngl` / `--n-gpu-layers` | **GPU에 오프로드할 레이어 수** |
| `-t` / `--threads` | CPU 스레드 수 |

```bash
# 서버 모드 — OpenAI 호환 API + Web UI
nohup build/bin/llama-server \
    -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --host 0.0.0.0 --port 8081 -ngl 99 > llama-server.log 2>&1 &
# → "server is listening on http://0.0.0.0:8081"
```

```python
# 기존 OpenAI SDK 코드에서 base_url만 바꿔 호출
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8081/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="local-model",     # 로컬 서버에선 임의 값 (실제 모델은 -m으로 결정)
    messages=[{"role": "user", "content": "어댑터즈를 한 문장으로 소개해줘"}],
    max_tokens=200,
)
```

```bash
# -hf 옵션: 다운로드 + 로드를 한 번에 (<user>/<repo>:<양자화태그>)
build/bin/llama-completion -hf Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M -p "어댑터즈는" -n 128 -ngl 99
```

### GPU 오프로드 (`-ngl`)

큰 모델은 전체 가중치를 GPU에 못 올림 → **지정한 수만큼의 레이어만 GPU에서 처리**, 나머지는 CPU

```
입력 토큰 → 레이어 0..k (GPU, -ngl 지정분) → 레이어 k+1.. (CPU) → 출력
```

> **주의**: GPU와 CPU가 각자 결과를 만들어 합치는 게 아님 — Transformer 레이어는 여전히 **순서대로** 처리되고, 각 레이어를 **어디서 계산하느냐**만 달라짐

| 설정 | 의미 |
|---|---|
| `-ngl 0` | GPU 미사용, CPU 중심 |
| `-ngl 10` | 일부 레이어만 GPU |
| `-ngl 99` | 가능한 많은 레이어를 GPU |

> `-ngl` 크게 준다고 항상 좋은 건 아님 — 메모리 초과 시 `CUDA out of memory`. **작은 값부터 시작해 점진적으로 올리기**. 적정값은 모델 크기·양자화 타입·GPU 메모리·컨텍스트 길이·KV 캐시에 좌우

### 한계

로컬 추론·엣지·소규모 테스트엔 강하지만, **대규모 프로덕션 서빙 엔진은 아님**

> 수십~수백 동시 요청을 안정적으로 처리하고 GPU 자원을 최대 활용해 높은 처리량을 내려면 **vLLM·TensorRT-LLM** 같은 서빙 특화 엔진이 적합. llama.cpp는 로컬 개발 서버·경량 내장 서버에 가까움
