# 4-1. LLM 배포 (Docker Containerization · MIG · Monitoring)

검증된 LLM 서빙 환경을 **컨테이너로 고정해 배포**하고, **GPU를 나눠 쓰며(MIG)**, 운영 중 상태를 **관측(Monitoring)** 하는 배포·운영 단계

> 3장이 "추론을 어떻게 빠르게"였다면, 여기는 "그 서버를 실제 인프라에 어떻게 올리고 운영하나"

---

## Docker Containerization

Docker로 **LLM 서빙 앱과 실행 환경을 컨테이너 이미지로 묶고**, GPU가 할당된 컨테이너로 배포·실행하는 과정

> 소스 코드만이 아니라 **실행 환경(CUDA·Python·vLLM·라이브러리·실행 명령) 전체를 하나의 실행 단위로 고정** — 일반 웹 컨테이너와 달리 GPU 연결·큰 모델 가중치가 관건

### 컨테이너화 핵심 요소

| 요소 | 설명 | 예시 |
|---|---|---|
| 베이스 이미지 | 기본 실행 환경 (CUDA 런타임 포함) | `nvidia/cuda:12.1.0-runtime-ubuntu22.04` |
| Python·라이브러리 | 서버 실행 의존성 | `vllm`, `torch`, `transformers` |
| 서빙 엔진 | API 서버 실행 구성 요소 | vLLM, TGI |
| GPU 연결 환경 | 호스트 NVIDIA GPU를 컨테이너에서 사용 | `--gpus all`, `nvidia-container-toolkit` |
| 모델 가중치 | 추론용 모델 파일 | 이미지 포함 or 볼륨 연결 |
| 실행 명령 | 컨테이너 시작 시 실행 | `vllm serve ...` |

> **모델 가중치는 이미지에 넣기보다 볼륨 연결이 일반적** — 용량이 커 이미지에 넣으면 빌드·배포가 느려짐. 이미지는 가볍게, 실행 시 모델 디렉토리만 연결. (네트워크 제한 환경·완전한 배포 단위가 필요하면 이미지에 포함)

### 사용 이유 — 환경 재현·스케일 아웃·롤백

- LLM 서버는 CUDA·PyTorch·vLLM·Transformers **버전 조합에 민감** → 서버마다 설치 상태가 조금만 달라도 오류·성능 차이 → 이미지로 고정해 **모든 노드 동일 재현**
- **스케일 아웃**: GPU 노드 추가 후 같은 이미지 실행 → 동일 서빙 인스턴스 빠르게 증설
- **롤백**: 새 이미지 오류 시 이전 검증 태그로 되돌림

### 사용 방법 — 작성 → 빌드 → 실행

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04           # CUDA 런타임 포함 베이스
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install vllm                                  # 서빙 엔진 설치
EXPOSE 8000                                            # 포트 사용 표시(실제 연결은 run에서 -p)
# 모델은 실행 시 /models로 볼륨 마운트한다고 가정
CMD ["vllm", "serve", "/models/Qwen2.5-1.5B-Instruct", "--port", "8000", \
     "--gpu-memory-utilization", "0.9", "--max-model-len", "4096"]
```

```bash
docker build -t vllm-server .                          # Dockerfile로 이미지 빌드 (-t=이름, .=빌드 컨텍스트)

docker run --gpus all -p 8000:8000 \                   # GPU 할당 + 포트 연결
    -v /host/models:/models \                          # 호스트 모델 디렉토리를 컨테이너 /models로
    vllm-server
```

| 옵션 | 역할 |
|---|---|
| `--gpus all` | 호스트 NVIDIA GPU를 컨테이너에 할당 (없으면 GPU 추론 불가) |
| `-p 8000:8000` | 호스트↔컨테이너 포트 연결 (외부 접근용) |
| `-v /host/models:/models` | 큰 모델 가중치를 이미지 밖에서 마운트 |

> **빌드 레이어 캐시**: Docker는 각 단계를 레이어로 저장 → 바뀌지 않은 단계는 재사용해 재빌드가 빠름
>
> macOS Docker Desktop은 `--gpus all` 미지원 → 빌드까지만 로컬, 실행은 **NVIDIA GPU가 있는 CUDA 호스트**에서

### 공식 이미지로 바로 서빙

실무에선 Dockerfile 작성·빌드를 건너뛰고 **vLLM 공식 이미지**를 받아 바로 실행하는 경우도 많음

```bash
docker run --gpus all -p 8000:8000 \
    -e VLLM_USE_V1=0 -e VLLM_ATTENTION_BACKEND=XFORMERS \  # T4 호환 V0+XFormers
    --shm-size=2g \                                         # 공유 메모리↑ (멀티프로세스·GPU 통신용, 부족 시 오류)
    --ipc=host \                                            # 호스트 IPC 사용 (멀티 GPU 통신 안정화)
    -v /host/models:/models \
    vllm/vllm-openai:v0.9.1 \                               # 공식 이미지 (Python·vLLM 사전 포함)
    --model /models/Qwen2.5-1.5B-Instruct --max-model-len 4096   # 이미지 뒤 인자는 vLLM 서버 옵션
```

> 이미지 이름 뒤 인자(`--model`·`--max-model-len`)는 Docker 옵션이 아니라 **컨테이너 안 vLLM 서버에 전달**되는 옵션

## Multi-Instance GPU (MIG)

NVIDIA GPU 하나를 **자체 연산 자원·메모리를 가진 여러 독립 GPU 인스턴스로 분할**하는 하드웨어 수준 파티셔닝 기술

> 단순 GPU 시간 분할이 아니라 **SM·VRAM·캐시를 인스턴스별로 실제 분리** → 각 인스턴스가 독립 GPU처럼 동작, 서로 침범 불가. 물리 서버를 여러 VM으로 나누는 것과 유사

> **SM (Streaming Multiprocessor)**: GPU 안 실제 연산 유닛 (CPU 코어에 해당) / **VRAM**: 모델 가중치·KV Cache·중간 결과를 올리는 GPU 전용 메모리

### vs 소프트웨어 GPU 공유

- 소프트웨어 공유: 여러 프로세스가 GPU 한 장을 함께 사용 (경계 없음)
- **MIG: 하드웨어 수준 분할로 인스턴스별 경계** → 한 인스턴스 작업이 다른 인스턴스 자원 침범 불가

### MIG 프로파일 (A100 80GB 예시)

프로파일 = 인스턴스 하나가 가져갈 GPU 자원 크기. 이름 `{N}g.{M}gb`에서 **`Ng`=자원 칸 수(인스턴스 개수 아님), `Mgb`=VRAM 크기**

| 프로파일 | 인스턴스 수 | 인스턴스당 VRAM | SM 수 | 이해 |
|---|---|---|---|---|
| `1g.10gb` | 7 | ~10GB | 14 | 작은 인스턴스 7개 |
| `2g.20gb` | 3 | ~20GB | 28 | 중간 3개 |
| `3g.40gb` | 2 | ~40GB | 42 | 큰 인스턴스 2개 |
| `7g.80gb` | 1 | 80GB | 98 | GPU 한 장 거의 그대로 |

> **`3g.40gb`가 왜 2개?** — `3g`는 "3개"가 아니라 인스턴스 하나가 가져가는 자원 크기(3칸+~40GB). A100 80GB는 이 크기를 2개까지 생성. 반대로 `1g.10gb`는 작아서 7개까지
>
> 프로파일 목록 확인: `nvidia-smi mig -lgip` (List GPU Instance Profiles) — GPU 모델·드라이버에 따라 이름·값이 달라짐

### 사용 이유 — 유휴 자원↓·격리·비용 효율

- 소형/양자화 모델은 A100·H100 일부 자원만 써도 충분 → GPU 통째 배정 시 자원 낭비 → **여러 인스턴스로 나눠 여러 모델·서비스 동시 서빙**
- **서비스 격리**: 한 서비스에 트래픽·메모리 부족이 나도 다른 인스턴스는 영향↓ → 여러 팀·모델·테스트가 한 GPU를 예측 가능하게 공유

### 사용 방법 — 호스트 분할 + 컨테이너 할당

모델·vLLM 코드는 그대로. 호스트에서 GPU를 인스턴스로 나누고, 컨테이너에 특정 인스턴스를 지정

| 구분 | 하는 일 |
|---|---|
| 호스트 서버 | MIG 모드 켜고 원하는 크기의 인스턴스 생성 |
| 컨테이너 실행 | `docker run --gpus '"device=0:0"'`로 특정 인스턴스 연결 |

```bash
sudo nvidia-smi -i 0 -mig 1              # ① 0번 GPU MIG 모드 활성화 (나눌 수 있는 상태로)
sudo nvidia-smi mig -i 0 -cgi 9,9 -C     # ② 프로파일 ID 9(3g.40gb) 인스턴스 2개 생성 (-C=Compute Instance까지)
nvidia-smi mig -i 0 -lgi                 # ③ 생성된 인스턴스 목록 확인

# ④ 첫 번째 인스턴스(0:0)에 vLLM 컨테이너
docker run --gpus '"device=0:0"' -p 8000:8000 -v /models:/models \
    vllm/vllm-openai:latest --model /models/Qwen2.5-1.5B-Instruct
# ⑤ 두 번째 인스턴스(0:1)에 다른 모델, 호스트 포트만 8001로 구분
docker run --gpus '"device=0:1"' -p 8001:8000 -v /models:/models \
    vllm/vllm-openai:latest --model /models/Mistral-7B-Instruct-v0.3
```

```
물리 GPU 0
├─ MIG 0:0 → 컨테이너 1 (Qwen)    → 호스트 8000
└─ MIG 0:1 → 컨테이너 2 (Mistral) → 호스트 8001
   두 컨테이너는 같은 GPU지만 서로 다른 인스턴스 → 한쪽 OOM·트래픽이 다른 쪽에 영향↓
```

> GPU 전체는 `device=0`, MIG는 `device=0:0`처럼 **`GPU번호:인스턴스번호`**. 컨테이너 안 포트는 둘 다 8000이지만 호스트에서 8000·8001로 구분
>
> **MIG 지원 GPU**: A100·A30·H100 등 **데이터센터용만** — RTX 3090·4090 같은 소비자용은 미지원. MIG 지원 GPU + NVIDIA 데이터센터 드라이버 + 호스트 관리자 권한(`sudo`) 필요

## Monitoring

운영 중인 LLM 서빙 서버의 **처리량·지연시간·GPU 메모리·대기 큐** 같은 지표를 지속 수집·시각화해 서버 상태를 관측하는 것

> 순간 확인이 아니라 **시간 축으로 변화를 관측** — "지금 얼마 쓰나"가 아니라 "평소보다 높아졌나, 언제부터, 트래픽과 함께 변했나"를 봄

### 관측 대상 지표

| 관측 대상 | 확인할 수 있는 신호 |
|---|---|
| 처리량 | 지금 얼마나 많은 요청·토큰을 처리 중인지 |
| 지연시간 | 사용자가 응답을 느리게 체감하는지 |
| GPU 메모리 사용률 | 모델·KV Cache 담을 여유가 남았는지 |
| KV Cache 사용률 | 긴 문맥·동시 요청으로 메모리 포화되는지 |
| 실행 중 요청 수 | 서버가 실제로 얼마나 바쁜지 |
| 대기 중 요청 수 | 용량 부족으로 요청이 밀리는지 |
| 에러 수 | 서버 오류·OOM·요청 실패가 발생하는지 |

### 관측성(Observability)의 세 요소

Monitoring은 **관측성** — 시스템 내부 상태를 외부에서 파악하는 성질 — 의 일부. 이 장은 **메트릭 중심**

| 요소 | 데이터 | 대표 질문 |
|---|---|---|
| **메트릭** (Metrics) | 시간에 따라 변하는 숫자 지표 | 지금 요청이 얼마나 몰렸고 GPU 메모리는? |
| 로그 (Logs) | 이벤트마다 남기는 텍스트 기록 | 방금 실패한 요청의 에러 메시지는? |
| 트레이스 (Traces) | 한 요청이 거친 단계별 경로·소요 시간 | 인증·검색·추론·응답 중 어디서 느려졌나? |

### vLLM 핵심 지표 (`/metrics`)

vLLM은 별도 옵션 없이 `/metrics`로 **Prometheus 형식** 지표 노출

| 지표 | 의미 | 타입 |
|---|---|---|
| `vllm:num_requests_running` | 현재 처리 중 요청 수 | gauge |
| `vllm:num_requests_waiting` | 대기 큐에 쌓인 요청 수 | gauge |
| `vllm:gpu_cache_usage_perc` | KV Cache가 쓰는 GPU 캐시 비율 (0~1) | gauge |
| `vllm:time_to_first_token_seconds` | 요청별 TTFT 분포 (`_bucket`·`_sum`·`_count`) | histogram |
| `vllm:generation_tokens_total` | 누적 생성 출력 토큰 수 | counter |
| `vllm:request_success_total` | 정상 완료 요청 수 | counter |

> **지표 타입**: `counter`=계속 누적(생성 토큰 총합), `gauge`=현재 상태(처리 중 요청 수), `histogram`=분포(TTFT를 구간별로)
>
> **벤치마크 vs Monitoring**: 벤치마크는 특정 조건에서 성능을 **한 번 측정하는 시험**, Monitoring은 운영 서버 상태를 **계속 확인하는 계기판** — 대상은 비슷하나 보는 방식이 다름

### 사용 이유 — 장애 조기 대응

장애 전 신호가 먼저 나타남 → 사용자가 "느려요" 하기 전에 지표로 원인 파악·조치:

- **대기 요청 수↑** = 들어오는 요청을 제때 못 처리 / **KV Cache 사용률 한계** = 곧 OOM 위험 / **TTFT 급증** = 첫 응답 지연
- 지표 기준으로 판단: 요청 더 받을 수 있나 / 동시 요청 한도 줄일까 / 모델 축소·GPU 증설할까
- TTFT·처리량·에러율로 **SLA 준수 확인**, 특정 시간대 반복 지연 시 트래픽 패턴에 맞춰 조정

### 사용 방법 — `/metrics` → Prometheus → 시각화

| 순서 | 단계 | 구성 요소 | 역할 |
|---|---|---|---|
| 1 | 지표 노출 | vLLM `/metrics` | 서버 상태 지표를 외부에서 읽게 제공 |
| 2 | 지표 수집 | **Prometheus** | `/metrics`를 주기적으로 읽어 시계열로 저장 |
| 3 | 지표 시각화 | Grafana (실무) / matplotlib (실습) | 저장된 지표를 그래프·대시보드로 |

> **Prometheus**: 서버가 내보내는 숫자 지표를 주기적으로 수집해 시간 순서로 저장하는 모니터링 도구
> **Grafana**: Prometheus 지표를 대시보드·알림으로 보여주는 시각화 도구. 실무는 **Prometheus 수집 + Grafana 대시보드** 조합이 표준 (실습은 Colab 포트 연결이 번거로워 matplotlib로 대체)

```bash
# 워밍업 요청 후 /metrics 조회 (트래픽이 흘러야 지표가 채워짐)
curl -s http://localhost:8000/metrics | grep -E "^vllm:"
```

> 누적 지표(`*_total`)는 처리 기록이 남지만, 현재 상태 gauge(`num_requests_running`·`gpu_cache_usage_perc`)는 요청 완료 후 조회하면 `0.0`으로 보임 → 부하 중에 관측해야 변화가 드러남
