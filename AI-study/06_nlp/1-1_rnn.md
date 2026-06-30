# 1-1. RNN (Recurrent Neural Network)

이전 시점의 hidden state를 다음 시점 계산에 재사용하며 순차 데이터를 처리하는 순환 구조의 인공 신경망

> 피드포워드 신경망은 이전 정보를 기억하지 못함 — RNN은 순환 연결로 시퀀스 문맥을 유지

---

## 구조

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

| 기호 | 설명 |
|---|---|
| $x_t$ | 현재 시점의 입력 벡터 |
| $h_{t-1}$ | 이전 시점의 hidden state |
| $h_t$ | 현재 시점의 hidden state (과거 문맥 누적) |
| $W_x$ | 입력에 적용되는 가중치 행렬 |
| $W_h$ | 이전 hidden state에 적용되는 가중치 행렬 |
| $b$ | 편향 |

| 개념 | 설명 |
|---|---|
| **셀 (Cell)** | 각 시점(t)마다 $x_t$와 $h_{t-1}$을 받아 $h_t$를 출력하는 최소 계산 단위 |
| **레이어 (Layer)** | 셀들이 시간 축으로 반복되는 한 층 |

## 핵심 구성 요소

### 입력 벡터 ($x_t$)

특정 시점 t에서 모델에 입력되는 벡터 — 신경망은 문자열을 직접 처리하지 못하므로 수치 벡터로 변환 필요

```
텍스트 → 토크나이징 → 인덱싱 → 임베딩 → x_t
"hello"     h,e,l,l,o   [7,4,11,11,14]   실수 벡터
```

- $x_t$의 차원 = `embedding_dim` (사용자가 설정하는 하이퍼파라미터)
- 배치 처리 시 전체 입력 텐서 shape: `(batch_size, seq_len, embedding_dim)` (`batch_first=True` 기준)

### 은닉 상태 ($h_t$, Hidden State)

이전 시점까지의 입력 정보를 요약해 담고 있으며, 다음 시점 계산에 전달되는 내부 상태 벡터

- `hidden_size`는 사용자가 설정하는 하이퍼파라미터 (64, 128, 256, 512 등)
- 초기 hidden state $h_0$는 0 벡터로 초기화

| 구분 | RNN | LSTM |
|---|---|---|
| 상태 종류 | $h_t$ 하나 | $h_t$ (단기) + $C_t$ (장기) 분리 |
| 장기 의존성 | 약함 | 강함 |

### 출력 벡터 ($y_t$)

$$y_t = W_y \cdot h_t + b_y$$

은닉 상태를 출력 레이어(Linear)에 통과시켜 과업에 맞는 차원으로 변환한 최종 예측 벡터

| 출력 유형 | 구조 | 예시 태스크 |
|---|---|---|
| Many-to-One | 마지막 $h_t$만 사용 | 감정 분류, 문서 분류 |
| Many-to-Many | 모든 시점의 $y_t$ 사용 | 품사 태깅, 기계 번역 |

> $y_t$의 차원은 과업에 따라 결정 — 다음 단어 예측: `vocab_size` / 분류: 클래스 수

### 순환 연결 (Recurrent Connection)

현재 시점의 $h_t$가 다음 시점에서 $h_{t-1}$로 재사용되는 구조 — RNN이 "순환"인 이유

```
t=1: h0(0벡터) + x1("Startupcode") → h1
t=2: h1        + x2("is")          → h2  ← h1 재사용
t=3: h2        + x3("a")           → h3  ← h2 재사용
t=4: h3        + x4("company.")    → h4  ← h3 재사용
```

| 구분 | 피드포워드 | 순환 연결 |
|---|---|---|
| 정보 흐름 | 입력→출력 (단방향) | 이전 $h_t$ → 현재 계산에 재사용 |
| 문맥 유지 | 불가 | 시간 축을 따라 과거 정보 누적 |

### 가중치 행렬 (Weight Matrices)

입력·상태 벡터를 새로운 표현 공간으로 선형 변환하기 위해 학습되는 파라미터 행렬

| 행렬 | 역할 | 형태 |
|---|---|---|
| $W_x$ | 현재 입력 $x_t$를 hidden 공간으로 변환 | `(hidden_size, embedding_dim)` |
| $W_h$ | 이전 상태 $h_{t-1}$의 문맥을 현재에 반영 | `(hidden_size, hidden_size)` |
| $W_y$ | hidden state $h_t$를 출력 공간으로 변환 | `(output_size, hidden_size)` |

> **가중치 공유**: 모든 시점에서 동일한 $W_x$, $W_h$, $W_y$를 재사용 → 시퀀스 길이와 무관하게 파라미터 수 고정

## 사용 이유

| 항목 | 설명 |
|---|---|
| 기억 능력 | 이전 시점의 정보를 기억해 현재 출력에 반영 |
| 시퀀스 데이터 처리 | 텍스트, 음성, 시계열 등 순차 데이터의 패턴·종속성 학습 |
| 시간적 패턴 학습 | 데이터 간 시간적 관계를 학습해 예측 성능 향상 |
| 다양한 응용 | 텍스트 분류, 감정 분석, 기계 번역, 음성 인식 등 |

## 학습 메커니즘

### 순전파 (Forward Pass)

입력 시퀀스를 시간 순서대로 처리하며 각 시점의 $h_t$를 계산하는 과정

$$h_t = f(W_x x_t + W_h h_{t-1} + b_h), \quad y_t = W_y h_t + b_y$$

```python
output, hidden = rnn(embeds)
# output: (batch, seq_len, hidden_size) — 모든 시점의 h_t
# hidden: (1, batch, hidden_size)       — 마지막 시점의 h_t
```

| 반환값 | 설명 | 사용 사례 |
|---|---|---|
| `output` | 모든 시점의 hidden state | 품사 태깅, 기계 번역 (Many-to-Many) |
| `hidden` | 마지막 시점의 hidden state | 감정 분류, 문서 분류 (Many-to-One) |

### 시간 역전파 (BPTT, Backpropagation Through Time)

RNN을 시간 축으로 펼친 뒤, 손실의 기울기를 과거 시점까지 전파해 공유 가중치를 업데이트하는 학습 알고리즘

```python
optimizer.zero_grad()
output, hidden = rnn(x)
loss = criterion(fc(output), target)
loss.backward()   # BPTT 자동 수행
optimizer.step()
```

| 구분 | 일반 역전파 | BPTT |
|---|---|---|
| 전파 방향 | 층 방향 (깊이 방향) | 층 방향 + 시간 방향 |
| 기울기 소실 위험 | 층이 깊을수록 커짐 | 시퀀스가 길수록 커짐 |

## 한계: 기울기 소실 (Vanishing Gradient)

역전파 시 $\tanh'(x) \leq 1$이 시점 수만큼 반복 곱해지면서 기울기가 지수적으로 감소 → 초기 입력 정보가 학습에서 소실

- **장기 의존성 문제**: 앞부분 중요 정보가 뒷부분 예측에 충분히 반영되지 않음

| 현상 | 원인 | 결과 |
|---|---|---|
| 기울기 소실 (Vanishing Gradient) | 반복 곱셈으로 기울기 → 0 | 초기 시점 가중치 미업데이트, 장기 문맥 학습 불가 |
| 기울기 폭발 (Exploding Gradient) | 반복 곱셈으로 기울기 → ∞ | 가중치 발산, loss가 NaN |

해결책: **LSTM** (셀 상태 덧셈 경로) / **GRU** / 기울기 클리핑 (Gradient Clipping)

## 코드 예시 (PyTorch)

```python
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # x_t 생성
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)  # h_t 계산
        self.fc = nn.Linear(hidden_dim, output_dim)  # y_t 변환

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out)
```
