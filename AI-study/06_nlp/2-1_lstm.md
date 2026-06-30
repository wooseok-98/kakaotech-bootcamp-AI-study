# 2-1. LSTM (Long Short-Term Memory)

장기 의존성을 안정적으로 학습하도록 설계된 순환 신경망 — RNN의 기울기 소실 문제를 Cell State와 게이트 구조로 해결

> RNN은 $h_t$ 하나로 정보를 전달해 기울기 소실 발생 — LSTM은 $c_t$(장기)와 $h_t$(단기)를 분리해 장기 의존성 유지

---

## 구조

$$\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) \\
\tilde{c}_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}$$

| 기호 | 설명 |
|---|---|
| $f_t$ | Forget Gate — 이전 셀 상태를 얼마나 유지할지 (0~1) |
| $i_t$ | Input Gate — 새로운 정보를 얼마나 반영할지 (0~1) |
| $\tilde{c}_t$ | 후보 셀 상태 — 현재 입력에서 생성된 새 정보 후보 |
| $c_t$ | Cell State — 장기 기억, 덧셈으로 갱신 |
| $o_t$ | Output Gate — 현재 상태를 얼마나 출력할지 (0~1) |
| $h_t$ | Hidden State — 단기 기억, 현재 출력 |
| $\odot$ | element-wise 곱 (Hadamard product) |

## 게이트 동작 과정

| 단계 | 게이트 | 설명 |
|---|---|---|
| 1 | Forget Gate | 이전 $c_{t-1}$ 중 남길 부분 결정 — 1에 가까울수록 유지, 0에 가까울수록 삭제 |
| 2 | Input Gate + $\tilde{c}_t$ | 새로운 후보 정보를 얼마나 받아들일지 결정 |
| 3 | Cell State 업데이트 | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ — 이전 기억 + 새 정보 결합 |
| 4 | Output Gate | $c_t$를 tanh 통과시킨 후 출력 비율 결정 → $h_t$ 생성 |

## Cell State (셀 상태)

LSTM 내부에서 장기 정보를 유지하고 다음 시점으로 전달하는 내부 상태값

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

| 구분 | Cell State ($c_t$) | Hidden State ($h_t$) |
|---|---|---|
| 역할 | 장기 기억 | 단기 기억 · 현재 출력 |
| 갱신 방식 | 덧셈 중심 | 곱셈 + tanh |
| 기울기 흐름 | 안정적 (소실 완화) | 소실 위험 있음 |
| RNN 비교 | RNN에 없음 | RNN의 $h_t$와 유사 |

> **덧셈 갱신이 핵심**: RNN의 $h_t$는 매 시점 tanh 곱셈으로 갱신되어 기울기 소실 발생. Cell State는 덧셈 중심으로 갱신되어 기울기가 더 안정적으로 전달됨 → 장기 의존성 학습에 유리

## RNN vs LSTM 비교

| 구분 | RNN | LSTM |
|---|---|---|
| 내부 상태 | $h_t$ 하나 | $c_t$ (장기) + $h_t$ (단기) |
| 기울기 소실 | 심각 | 완화 |
| 장기 의존성 | 약함 | 강함 |
| 게이트 구조 | 없음 | 3개 (Forget / Input / Output) |
| 파라미터 수 | 적음 | 많음 |

## 학습 메커니즘

### 순전파 (Forward Pass)

입력 $x_t$, 이전 상태 $h_{t-1}$·$c_{t-1}$을 받아 게이트를 순서대로 계산해 $h_t$와 $c_t$ 생성

```
① 망각 게이트 (f_t) → ② 입력 게이트 (i_t) + 후보 셀 (c̃_t)
→ ③ 셀 상태 갱신 (c_t) → ④ 출력 게이트 (o_t) → ⑤ 은닉 상태 (h_t)
```

```python
output, (h_n, c_n) = lstm(embeds)
# output: (batch, seq_len, hidden_size) — 모든 시점의 h_t
# h_n:    (1, batch, hidden_size)       — 마지막 h_t
# c_n:    (1, batch, hidden_size)       — 마지막 c_t
```

| 구분 | RNN 순전파 | LSTM 순전파 |
|---|---|---|
| 입력 상태 | $h_{t-1}$ | $h_{t-1}$, $c_{t-1}$ |
| 계산 단계 | 1개 (tanh 연산) | 게이트 3 + 후보 + 갱신 + 출력 |
| 상태 갱신 | $h_t$만 | $h_t$와 $c_t$ 동시 |

### 시간 역전파 (BPTT)

손실에서 $h_t$·$c_t$ 경로를 따라 과거 시점으로 기울기를 전파해 가중치 갱신

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

- $f_t \approx 1$이면 기울기가 거의 감쇠 없이 과거로 전달 → 장기 의존성 학습 가능
- RNN BPTT: $\tanh'$ 반복 곱셈으로 기울기 소실 심각
- LSTM BPTT: $c_t$ 덧셈 경로로 기울기가 더 안정적으로 전달

```python
loss.backward()  # BPTT 자동 수행 — 모든 게이트 파라미터 기울기 자동 계산
```

> 기울기 폭발은 여전히 발생 가능 → `clip_grad_norm_(model.parameters(), max_norm=1.0)` 병행 사용

## 코드 예시 (PyTorch)

```python
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        # output: (batch, seq_len, hidden_size) — 모든 시점의 h_t
        # hidden: (1, batch, hidden_size)       — 마지막 h_t
        # cell:   (1, batch, hidden_size)       — 마지막 c_t
        return self.fc(output)
```
