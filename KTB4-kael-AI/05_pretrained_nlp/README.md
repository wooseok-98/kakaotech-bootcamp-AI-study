# 05_pretrained_nlp — 전이 학습 & NLP

사전 훈련된 모델을 활용한 전이 학습과 하이퍼파라미터 튜닝, 챗봇 제작 실습 과제입니다.

---

## 파일 구성

| 파일 | 과제 | 상태 |
|------|------|------|
| `01_resnet_classification.ipynb` | 과제 1 — ResNet 이미지 분류 | ✅ 완료 |
| `02_vgg16_resnet_comparison.ipynb` | 과제 2, 3 — VGG16 전이 학습 & 성능 비교 | ✅ 완료 |
| `03_hyperparameter_tuning.ipynb` | 과제 4 — 하이퍼파라미터 튜닝 | ✅ 완료 |

---

## 과제 목록

### 과제 1 — ResNet 이미지 분류 (`01_resnet_classification.ipynb`)
- **과제 1.** ResNet18 사전 학습 모델 로드 후 CIFAR-10 이미지 분류

### 과제 2, 3 — VGG16 전이 학습 & 성능 비교 (`02_vgg16_resnet_comparison.ipynb`)
- **과제 2.** VGG16 사전 학습 모델로 전이 학습 수행 (Feature Extraction 방식)
- **과제 3.** 동일 데이터셋에서 ResNet18과 VGG16 성능 비교

### 과제 4 — 하이퍼파라미터 튜닝 (`03_hyperparameter_tuning.ipynb`)
- **과제 4.** 가상 데이터셋 생성 후 GridSearch & RandomSearch로 하이퍼파라미터 탐색

### 과제 5 — 챗봇 제작 (미완료)
- **과제 5-1.** 문장 입력 시 다음 단어를 생성하는 언어 모델 구현
- **과제 5-2.** 모델 반복 호출로 완전한 문장 생성
- **과제 5-3.** FastAPI로 웹 API 배포

---

## 폴더 구조

```
05_pretrained_nlp/
├── README.md
├── 01_resnet_classification.ipynb    # 과제 1 — ResNet 이미지 분류
├── 02_vgg16_resnet_comparison.ipynb  # 과제 2, 3 — VGG16 전이 학습 & 성능 비교
└── 03_hyperparameter_tuning.ipynb    # 과제 4 — 하이퍼파라미터 튜닝
```

---

## 환경 설정

```bash
pip install torch torchvision datasets matplotlib tqdm scikit-learn
```

> CIFAR-10 데이터셋은 [HuggingFace Hub](https://huggingface.co/datasets/uoft-cs/cifar10)에서 자동 다운로드됩니다.
