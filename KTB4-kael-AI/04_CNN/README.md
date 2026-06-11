# 04_CNN — 머신러닝 알고리즘 및 CNN 이미지 분류

머신러닝 기초 알고리즘부터 딥러닝 CNN까지, PyTorch 기반으로 실습하는 과제입니다.

---

## 파일 구성

| 파일 | 과제 | 상태 |
|------|------|------|
| `01_dataset_knn.ipynb` | 과제 1, 2 — 데이터셋 분할 & K-NN 분류 | ✅ 완료 |
| `02_algorithm_comparison.ipynb` | 과제 3 — 4가지 분류 알고리즘 비교 | ✅ 완료 |
| `03_augmentation.ipynb` | 과제 4 — Data Augmentation 효과 비교 | ✅ 완료 |
| `04_activation_mlp.ipynb` | 과제 5, 6 — 활성화 함수 & MLP 분류 | ✅ 완료 |
| `05_cnn.ipynb` | 과제 7 — CNN 이미지 분류 (CIFAR-10) | ✅ 완료 |

---

## 과제 목록

### 과제 1, 2 — 데이터셋 분할 & K-NN (`01_dataset_knn.ipynb`)
- **과제 1.** 가상 데이터셋을 생성한 뒤, 학습·검증·테스트 데이터셋으로 분할
- **과제 2.** 가상 데이터셋에 K-최근접이웃(K-NN) 알고리즘으로 학습·예측 수행

### 과제 3 — 분류 알고리즘 비교 (`02_algorithm_comparison.ipynb`)
- **과제 3.** 동일한 이진 분류 데이터셋에 Perceptron, SVM, Random Forest, Naive Bayes를 적용해 성능 비교

### 과제 4 — Data Augmentation (`03_augmentation.ipynb`)
- **과제 4.** Data Augmentation 적용 전후 모델 성능 비교

### 과제 5, 6 — 활성화 함수 & MLP (`04_activation_mlp.ipynb`)
- **과제 5.** Sigmoid, Tanh, ReLU, Leaky ReLU 직접 구현 및 그래프 시각화
- **과제 6.** `make_moons` 비선형 데이터셋에 MLP 분류기 설계 및 학습
  - 결정 경계 시각화 + 정분류/오분류 강조 시각화

### 과제 7 — CNN 이미지 분류 (`05_cnn.ipynb`)
- **과제 7.** CNN을 직접 구성하여 CIFAR-10 이미지 분류 수행
  - 3단 Conv-ReLU-MaxPool 구조 + FC 분류기

---

## 폴더 구조

```
04_CNN/
├── README.md
├── 01_dataset_knn.ipynb          # 과제 1, 2 — 데이터셋 분할 & K-NN
├── 02_algorithm_comparison.ipynb # 과제 3 — 알고리즘 비교
├── 03_augmentation.ipynb         # 과제 4 — Data Augmentation
├── 04_activation_mlp.ipynb       # 과제 5, 6 — 활성화 함수 & MLP
├── 05_cnn.ipynb                  # 과제 7 — CNN 이미지 분류
└── data/                         # CIFAR-10 자동 다운로드 경로
```

---

## 회고

<details>
<summary>머신러닝 기초 알고리즘 (K-NN, SVM, Random Forest 등)</summary>

AI 연구를 하며 이론으로만 접했던 알고리즘들을 직접 코드로 구현하고 비교해보는 경험이 새로웠다. 특히 동일한 데이터셋에 Perceptron, SVM, Random Forest, Naive Bayes를 순서대로 적용하면서, 각 알고리즘이 데이터를 바라보는 관점이 얼마나 다른지 구체적으로 느낄 수 있었다. 단순히 정확도 숫자를 비교하는 것을 넘어, 모델마다 결정 경계의 모양과 특성이 다르다는 점이 인상적이었다.

</details>

<details>
<summary>활성화 함수 & MLP 분류</summary>

Sigmoid, Tanh, ReLU, Leaky ReLU를 직접 NumPy로 구현하고 그래프로 나란히 그려보니, 평소 "ReLU를 쓰면 좋다"고만 알고 있던 내용이 왜 그런지 시각적으로 이해됐다. Sigmoid의 기울기 소실 문제나 Leaky ReLU가 음수 영역을 살짝 열어두는 이유가 그래프 하나로 명확해졌다.

MLP 과제에서는 `make_moons`의 비선형 경계를 99% 정확도로 분류해냈는데, 결정 경계를 직접 시각화해보니 모델이 두 반달 모양을 따라 꽤 정교하게 경계를 그어놓은 것이 인상적이었다. 오분류된 4개의 점도 경계 근처에 몰려 있어, 모델이 어느 부분에서 헷갈리는지도 한눈에 파악할 수 있었다.

</details>

<details>
<summary>CNN 이미지 분류 (CIFAR-10)</summary>

처음에는 랜덤 노이즈 데이터로 학습했더니 정확도가 10% 내외에 머물렀는데, 이게 10클래스 무작위 추측 확률과 정확히 같다는 걸 깨닫고 실제 데이터셋인 CIFAR-10으로 교체했다. 모델 구조는 (3, 32, 32) 입력이라 변경 없이 바로 붙을 수 있었다.

10 에포크 학습 결과 train acc 94.9%, val acc 75.8%가 나왔다. 학습 곡선을 보면 에포크가 진행될수록 train loss는 계속 낮아지는데 val loss는 3~4 에포크 이후 다시 올라가는 전형적인 과적합 양상이다. Dropout이나 Data Augmentation 같은 정규화 기법 없이 단순 CNN으로 CIFAR-10에서 75%는 합리적인 수치지만, 다음에는 Batch Normalization이나 Augmentation을 추가해 과적합을 줄여보고 싶다.

정분류 샘플 이미지를 역정규화해서 원본 색상으로 복원해 시각화해보니, 모델이 실제로 어떤 이미지를 잘 맞히는지 눈으로 확인할 수 있어서 결과를 이해하는 데 도움이 됐다.

</details>
