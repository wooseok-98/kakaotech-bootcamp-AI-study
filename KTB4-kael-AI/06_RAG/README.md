# 06_RAG - Mini Project

공개 Wikipedia 문서를 대상으로 RAG 파이프라인을 구축하고, Gemini API와 FastAPI REST API로 질의응답을 제공하는 미니프로젝트입니다.

## 파일

| 파일 | 설명 |
| --- | --- |
| `rag_mini_project_colab.ipynb` | Colab 실행용 RAG 미니프로젝트 노트북 |

## 구현 범위

- Document Loading: Wikipedia 공개 문서 로딩
- Text Preprocessing: 유니코드 정규화, 공백/제어 문자 정리
- Chunking: word 기반 chunk size/overlap 분할
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Vector Store: FAISS `IndexFlatIP`
- Retrieval: 질문 임베딩 후 top-k 유사 청크 검색
- Generation: Gemini API 기반 한국어 답변 생성
- Serving: FastAPI `/health`, `/search`, `/query`
- Evaluation: RAGAS `context_recall`, `faithfulness`, `factual_correctness`

## Colab 실행 순서

1. `rag_mini_project_colab.ipynb`를 Colab에 업로드합니다.
2. Colab Secrets에 `GEMINI_API_KEY`를 저장합니다.
3. 위에서 아래로 셀을 실행합니다.
4. FastAPI 셀 실행 후 로컬 테스트 셀로 `/query` 응답을 확인합니다.
5. 필요하면 ngrok 셀 주석을 해제해 외부 URL을 발급합니다.
6. 마지막 RAGAS 평가 셀을 실행해 평가 결과를 확인합니다.

## 개인 프로젝트 확장 방향

미니프로젝트의 `fetch_wikipedia_extract()` 로더를 로컬 마크다운 로더로 교체하면 기존 `chatbot_project`에 강의 노트 Q&A용 RAG 모드를 붙일 수 있습니다.
