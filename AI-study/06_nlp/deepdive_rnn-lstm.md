# RNN vs LSTM — 텍스트 시퀀스 처리 차이

> **딥다이브 주제:** RNN과 LSTM이 텍스트 시퀀스 데이터를 처리하는 차이점을 설명하시오

---

## 들어가며

텍스트는 순서가 의미를 결정하는 데이터다.

- "나는 밥을 먹었다" ≠ "밥은 나를 먹었다"
- ANN·CNN은 입력을 독립적으로 처리 → 순서 의존성을 다루지 못함

RNN과 LSTM은 모두 "이전 정보를 기억하는 내부 상태"로 이 문제를 해결하려 한다.  
차이는 **어떻게 기억하느냐**다.

---

## 1. RNN의 텍스트 처리 방식

### 구조

![RNN unrolled](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

- 각 셀은 현재 입력 `x_t`와 이전 hidden state `h_t-1`을 받아 새로운 `h_t`를 생성
- `h_t` **하나에 현재까지의 모든 문맥 정보를 압축**해야 한다

### 텍스트 처리 예시

문장: `"The dog that chased the cat in the park was tired."`

- `t=2` "dog" → h₂에 "dog(단수 주어)" 정보가 저장됨
- `t=3~9` 관계절 8개 단어를 처리하는 동안 h_t가 계속 새 정보로 덮어씌워짐
- `t=10` "was" → 수일치를 위해 "dog가 단수"라는 정보가 필요한데, h₂의 내용이 8번의 변환을 거치는 사이 묻혀버린 상태

### 한계: 기울기 소실

학습할 때 "was에서 발생한 오차"를 역방향으로 전달해 "dog의 가중치"를 업데이트해야 하는데,  
매 시점을 거칠 때마다 tanh 미분값(≈ 0.5)이 곱해지면서 신호가 급격히 줄어든다.

```
"was"(t=10) → "dog"(t=2) 기울기 전달 과정

 t=10   t=9    t=8    t=7    t=6    t=5    t=4     t=3     t=2
 1.0  → 0.50 → 0.25 → 0.13 → 0.06 → 0.03 → 0.016 → 0.008 → 0.004
```

8번을 거치니 원래 오차의 **0.4%만 남는다.**  
이 미미한 신호로는 "dog"와 "was"의 수일치 관계를 학습할 수 없다.

---

## 2. LSTM의 텍스트 처리 방식

### 구조 — 두 가지 경로

![LSTM chain](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

RNN과 달리 LSTM은 **Cell State(장기 기억)와 Hidden State(단기 기억)** 두 경로를 유지한다.

![LSTM cell state highway](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-C-line.png)

| 경로 | 역할 | 갱신 방식 |
|---|---|---|
| Cell State `c_t` (위쪽) | 장기 기억 — 시퀀스 전체에 걸쳐 중요 정보 유지 | 덧셈 중심 |
| Hidden State `h_t` (아래쪽) | 단기 기억 — 현재 출력, 다음 게이트 입력 | 곱셈 + tanh |

### 게이트 3개의 역할

**망각 게이트 (Forget Gate)** — 이전 기억 중 무엇을 버릴까?

![Forget gate](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png)

**입력 게이트 (Input Gate)** — 새 정보를 얼마나 받아들일까?

![Input gate](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png)

**셀 상태 갱신 + 출력 게이트 (Output Gate)** — Cell State를 업데이트하고, 얼마나 꺼낼까?

![Output gate](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png)

### 텍스트 처리 예시 — 같은 문장

`"The dog that chased the cat in the park was tired."`

- `t=2` "dog" → 입력 게이트가 "dog(단수 주어)" 정보를 Cell State에 저장
- `t=3~9` 관계절 처리 중 → 망각 게이트가 f ≈ 1로 유지해 "dog 단수" 정보가 삭제되지 않음
- `t=10` "was" → Cell State에 "dog 단수" 정보가 그대로 남아있어 수일치 처리 가능 ✓

RNN은 h_t가 8번 덮어씌워지지만, LSTM은 Cell State에 "dog의 단수" 정보를 **의도적으로 보존**한다.

### 기울기 흐름 비교

같은 경로(`t=10` → `t=2`)로 기울기가 전달될 때:

```
        t=10   t=9   t=8   t=7   t=6   t=5   t=4   t=3   t=2
RNN     1.0   0.50  0.25  0.13  0.06  0.03  0.016 0.008 0.004  ← 0.4% 도달
LSTM    1.0   0.90  0.81  0.73  0.66  0.59  0.53  0.48  0.43   ← 43% 도달
```

- **RNN**: 매 시점 tanh 미분(≈ 0.5) 반복 곱 → 지수적 감소
- **LSTM**: Cell State 경로에서 망각 게이트(f ≈ 0.9)만 곱해짐 → 완만한 감소

망각 게이트가 "중요한 정보는 흘려보내지 않겠다"고 판단하면(f → 1), 기울기도 거의 감쇠 없이 전달된다.

---

## 3. RNN vs LSTM — 차이점

### 처리 방식의 근본 차이

| | RNN | LSTM |
|---|---|---|
| 새 토큰 입력 시 | 이전 `h_t`를 통째로 덮어씀 | 무엇을 버리고 추가할지 선택 |
| 갱신 방식 | `h_t = tanh(이전문맥 + 현재입력)` | `c_t = (이전기억 × 망각) + (새정보 × 입력)` |
| 결과 | 시간이 지나면 초반 정보 희석 | 게이트가 중요한 정보를 의도적으로 보존 |

### 종합 비교

| 구분 | RNN | LSTM |
|---|---|---|
| 메모리 구조 | `h_t` 하나 | `c_t`(장기) + `h_t`(단기) 분리 |
| 정보 선택 | 없음 — 모든 입력이 `h_t`에 반영 | 게이트 3개로 유지·추가·출력 선택 |
| 기울기 흐름 | tanh 반복 곱 → 지수적 감소 | Cell State 덧셈 경로 → 안정적 |
| 장기 의존성 | ~10 토큰 수준에서 한계 | 수백 토큰까지 처리 가능 |
| 파라미터 수 | 적음 | 약 4배 (게이트 3개 + 후보 셀) |
| 연산 속도 | 빠름 | 상대적으로 느림 |

### 태스크별 비교

| 태스크 | 의존 거리 | RNN | LSTM |
|---|---|---|---|
| 품사 태깅 | 짧음 | 충분 | 충분 |
| 감정 분류 (단문) | 짧음 | 가능 | 가능 |
| 주어-동사 수일치 | 중간~긺 | 불안정 | 안정적 |
| 기계 번역 | 긺 | 성능 저하 | 양호 |
| 장문 텍스트 생성 | 매우 긺 | 문맥 붕괴 | 상대적 안정 |

### 결론

RNN과 LSTM의 차이는 **어떻게 과거 정보를 관리하느냐**다.

- **RNN**: 하나의 상태에 모든 정보를 압축 → 오래된 정보가 자연스럽게 밀려남
- **LSTM**: 장기/단기를 분리하고 게이트로 선택 → 중요한 정보를 의도적으로 보존

이 구조적 차이가 기울기 흐름에서도 그대로 나타난다.  
RNN은 긴 시퀀스에서 **학습 자체**가 어렵고, LSTM은 Cell State 덕분에 기울기가 안정적으로 전달된다.

> **다음 단계**: Transformer의 Self-Attention은 순환 구조 없이 전체 시퀀스를 한 번에 참조하며 장기 의존성을 해결한다 — RNN/LSTM의 순차 처리 한계를 구조적으로 극복한 방식이다.

---

*이미지 출처: Christopher Olah, "Understanding LSTM Networks", 2015*
