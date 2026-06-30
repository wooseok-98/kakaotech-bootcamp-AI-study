# kakaotech-bootcamp-AI-study

카카오테크 부트캠프 AI 실무개발 과정에서 학습한 내용을 정리한 노트 모음입니다.
웹 기초부터 FastAPI, 데이터 분석·시각화, 머신러닝·딥러닝(CNN), NLP, 그리고 LLM 애플리케이션(RAG · LangChain · LangGraph)까지의 학습 흐름을 담았습니다.

---

## 목차

| 챕터 | 주제 |
| --- | --- |
| [01. 웹의 이해](#01_웹의-이해) | 클라이언트 - 서버 구조 |
| [02. FastAPI](#02_fastapi) | FastAPI · HTTP · REST API |
| [03. 데이터 분석](#03_데이터분석) | NumPy · Pandas |
| [04. 데이터 시각화](#04_데이터시각화) | Matplotlib · Seaborn · SciPy |
| [05. 인공지능 (CNN)](#05_인공지능cnn) | 머신러닝 · 딥러닝 · CNN · 전이 학습 |
| [06. NLP](#06_nlp) | RNN · LSTM |
| [07. RAG](#07_rag) | 검색 증강 생성 (Retrieval-Augmented Generation) |
| [08. LangChain](#08_langchain) | LLM 오케스트레이션 |
| [09. LangGraph](#09_langgraph) | LLM 에이전트 · 그래프 워크플로 |

---

## 01_웹의 이해

| 파일 | 주제 |
| --- | --- |
| [1-1_client-server.md](01_웹의%20이해/1-1_client-server.md) | Client - Server |

---

## 02_FastAPI

| 파일 | 주제 |
| --- | --- |
| [1-1_fastapi-setup.md](02_FastAPI/1-1_fastapi-setup.md) | FastAPI |
| [2-1_http.md](02_FastAPI/2-1_http.md) | HTTP |
| [2-2_request-data.md](02_FastAPI/2-2_request-data.md) | 요청 데이터 (Query String, Path Variable, JSON) |
| [3-1_rest-api.md](02_FastAPI/3-1_rest-api.md) | REST API |
| [4-1_design-pattern.md](02_FastAPI/4-1_design-pattern.md) | 디자인 패턴 |

---

## 03_데이터분석

| 파일 | 주제 |
| --- | --- |
| [1-1_numpy.md](03_데이터분석/1-1_numpy.md) | NumPy 기본 |
| [1-2_indexing.md](03_데이터분석/1-2_indexing.md) | 인덱싱 (Indexing) |
| [1-3_operations.md](03_데이터분석/1-3_operations.md) | 연산 & 유니버설 함수 |
| [2-1_pandas-series.md](03_데이터분석/2-1_pandas-series.md) | Pandas & 시리즈 (Series) |
| [2-2_dataframe.md](03_데이터분석/2-2_dataframe.md) | 데이터프레임 (DataFrame) |
| [3-1_filtering.md](03_데이터분석/3-1_filtering.md) | 필터링 (Filtering) |
| [3-2_grouping.md](03_데이터분석/3-2_grouping.md) | 그룹화 (Grouping) |
| [3-3_merging.md](03_데이터분석/3-3_merging.md) | 병합 (Merging) |
| [3-4_missing-duplicates.md](03_데이터분석/3-4_missing-duplicates.md) | 결측치 처리 & 중복 제거 |
| [3-5_pivot.md](03_데이터분석/3-5_pivot.md) | 피벗 (Pivot) |
| [3-6_string-operations.md](03_데이터분석/3-6_string-operations.md) | 문자열 처리 (String Operations) |
| [summary.md](03_데이터분석/summary.md) | 한 줄 정리 |

---

## 04_데이터시각화

| 파일 | 주제 |
| --- | --- |
| [1-1_데이터준비및변환.md](04_데이터시각화/1-1_데이터준비및변환.md) | 데이터 준비 및 변환 |
| [2-1_matplotlib.md](04_데이터시각화/2-1_matplotlib.md) | Matplotlib |
| [3-1_seaborn.md](04_데이터시각화/3-1_seaborn.md) | Seaborn |
| [4-1_시계열데이터.md](04_데이터시각화/4-1_시계열데이터.md) | 시계열 데이터 (리샘플링 · 이동평균 · 금융 데이터) |
| [5-1_scipy통계분석.md](04_데이터시각화/5-1_scipy통계분석.md) | SciPy 통계 분석 (정규분포 · 가설검정) |

---

## 05_인공지능(CNN)

| 파일 | 주제 |
| --- | --- |
| [0-0_summary.md](05_인공지능(CNN)/0-0_summary.md) | 한 줄 정리 |
| [1-1_intro.md](05_인공지능(CNN)/1-1_intro.md) | TensorFlow · Keras · PyTorch · Kaggle |
| [1-2_math-vector.md](05_인공지능(CNN)/1-2_math-vector.md) | 수학적 함수 · 벡터 |
| [2-1_data-preprocessing.md](05_인공지능(CNN)/2-1_data-preprocessing.md) | 데이터 전처리 · 데이터 증강 · 데이터셋 분할 |
| [3-1_machine-learning.md](05_인공지능(CNN)/3-1_machine-learning.md) | Machine Learning (머신 러닝) |
| [3-2_ml-algorithms.md](05_인공지능(CNN)/3-2_ml-algorithms.md) | Random Forest · K-NN · SVM · Naive Bayes |
| [4-1_deep-learning.md](05_인공지능(CNN)/4-1_deep-learning.md) | Deep Learning · Perceptron |
| [4-2_neural-network.md](05_인공지능(CNN)/4-2_neural-network.md) | Activation Function · ANN · FCNN · FCL |
| [4-3_training.md](05_인공지능(CNN)/4-3_training.md) | Loss Function · Backpropagation · Optimizer |
| [4-4_cnn.md](05_인공지능(CNN)/4-4_cnn.md) | CNN · Convolutional/Pooling/Flatten Layer · Model Architecture |
| [5-1_pretrained-models.md](05_인공지능(CNN)/5-1_pretrained-models.md) | 사전 학습 모델 (Pretrained Models) |
| [5-2_transfer-learning.md](05_인공지능(CNN)/5-2_transfer-learning.md) | 전이 학습 (Transfer Learning) |
| [6-1_model-tuning.md](05_인공지능(CNN)/6-1_model-tuning.md) | 모델 튜닝 (하이퍼파라미터 · 정규화) |
| [7-1_nlp.md](05_인공지능(CNN)/7-1_nlp.md) | 자연어 처리 (NLP) 개요 |

---

## 06_nlp

| 파일 | 주제 |
| --- | --- |
| [1-1_rnn.md](06_nlp/1-1_rnn.md) | RNN (순환 신경망) |
| [2-1_lstm.md](06_nlp/2-1_lstm.md) | LSTM |
| [deepdive_rnn-lstm.md](06_nlp/deepdive_rnn-lstm.md) | RNN · LSTM Deep Dive |

---

## 07_RAG

| 파일 | 주제 |
| --- | --- |
| [7-1_RAG_overivew.md](07_RAG/7-1_RAG_overivew.md) | RAG 개요 |
| [7-2_벡터_유사도_계산.md](07_RAG/7-2_벡터_유사도_계산.md) | 벡터 유사도 계산 |
| [7-3_Chunking_요약.md](07_RAG/7-3_Chunking_요약.md) | Chunking (청킹) |
| [7-4_Data_Input.md](07_RAG/7-4_Data_Input.md) | 데이터 입력 (Data Input) |
| [7-5_Document_Preprocessing_and_Indexing.md](07_RAG/7-5_Document_Preprocessing_and_Indexing.md) | 문서 전처리 및 인덱싱 |
| [7-6_Retrieval_Generation_ContextWindow_PromptEngineering.md](07_RAG/7-6_Retrieval_Generation_ContextWindow_PromptEngineering.md) | 검색 · 생성 · 컨텍스트 윈도우 · 프롬프트 엔지니어링 |
| [summary.md](07_RAG/summary.md) | 한 줄 정리 |

---

## 08_LangChain

| 파일 | 주제 |
| --- | --- |
| [1-1_langchain-basics.md](08_LangChain/1-1_langchain-basics.md) | LangChain 기초 |
| [2-1_prompt-templates.md](08_LangChain/2-1_prompt-templates.md) | 프롬프트 템플릿 |
| [3-1_structured-output.md](08_LangChain/3-1_structured-output.md) | 구조화된 출력 (Structured Output) |
| [4-1_lcel-runnables.md](08_LangChain/4-1_lcel-runnables.md) | LCEL · Runnable |
| [5-1_message-history.md](08_LangChain/5-1_message-history.md) | 메시지 히스토리 |
| [6-1_document-loaders.md](08_LangChain/6-1_document-loaders.md) | 문서 로더 (Document Loaders) |
| [7-1_text-splitters.md](08_LangChain/7-1_text-splitters.md) | 텍스트 분할 (Text Splitters) |
| [8-1_embeddings-vectorstore.md](08_LangChain/8-1_embeddings-vectorstore.md) | 임베딩 · 벡터 스토어 |
| [9-1_retrievers.md](08_LangChain/9-1_retrievers.md) | 리트리버 (Retrievers) |
| [10-1_langsmith.md](08_LangChain/10-1_langsmith.md) | LangSmith |
| [summary.md](08_LangChain/summary.md) | 한 줄 정리 |

---

## 09_LangGraph

| 파일 | 주제 |
| --- | --- |
| [1-1_ai-agent.md](09_LangGraph/1-1_ai-agent.md) | AI 에이전트 |
| [2-1_graph-structure.md](09_LangGraph/2-1_graph-structure.md) | 그래프 구조 |
| [3-1_langgraph-core.md](09_LangGraph/3-1_langgraph-core.md) | LangGraph 핵심 |
| [4-1_persistence.md](09_LangGraph/4-1_persistence.md) | 영속성 (Persistence) |
| [4-2_durable-execution.md](09_LangGraph/4-2_durable-execution.md) | 지속 실행 (Durable Execution) |
| [4-3_memory.md](09_LangGraph/4-3_memory.md) | 메모리 (Memory) |
| [4-4_human-in-the-loop.md](09_LangGraph/4-4_human-in-the-loop.md) | Human-in-the-Loop |
| [4-5_agent-patterns.md](09_LangGraph/4-5_agent-patterns.md) | 에이전트 패턴 |
| [2026_06_28_LangGraph.ipynb](09_LangGraph/2026_06_28_LangGraph.ipynb) | LangGraph 실습 노트북 |
| [summary.md](09_LangGraph/summary.md) | 한 줄 정리 |
