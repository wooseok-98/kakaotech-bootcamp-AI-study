| 개념                         | 한 줄 정리                                                   |
| -------------------------- | -------------------------------------------------------- |
| RAG                        | 외부 문서를 검색(Retrieval)하여 LLM 입력에 추가한 뒤 근거 기반 답변을 생성하는 아키텍처 |
| Tokenization               | 텍스트를 모델이 처리할 수 있는 토큰 단위로 분할하는 과정                         |
| Text Embedding             | 텍스트를 의미를 보존한 고정 길이 벡터로 변환하는 기법                           |
| Dense Vector               | 문맥적 의미를 반영한 밀집 벡터로 의미 기반 검색에 사용                          |
| Sparse Vector              | 단어 출현 정보를 기반으로 한 희소 벡터로 키워드 검색에 사용                       |
| Dot Product                | 두 벡터의 대응 원소를 곱해 더하여 유사성을 계산하는 연산                         |
| Cosine Similarity          | 벡터 크기를 제거하고 방향만 비교하여 유사도를 계산하는 방법                        |
| Vector Database | 임베딩 벡터를 저장하고 유사도 검색을 수행하는 데이터베이스 |
| ANN | 최근접 벡터를 근사적으로 빠르게 찾는 검색 기법 |
| HNSW | 계층형 그래프 기반 ANN 인덱스 |
| ChromaDB | 벡터·문서·메타데이터를 함께 관리하는 벡터 DB |
| Pinecone | 클라우드 기반 관리형 벡터 DB |
| FAISS | 대규모 벡터 검색을 위한 고성능 라이브러리 |
| Chunking                   | 긴 문서를 검색과 임베딩에 적합한 크기의 청크로 분할하는 과정                       |
| Fixed-size Chunking        | 미리 정한 크기 기준으로 문서를 일정하게 분할하는 방식                           |
| Regex & Delimiter Chunking | 줄바꿈·제목·구분자를 기준으로 문서를 분할하는 방식                             |
| Semantic Chunking          | 문장 간 의미 유사도를 이용해 주제 단위로 분할하는 방식                          |
| Parent-Child Chunking      | 검색용 작은 청크와 문맥용 큰 청크를 함께 관리하는 방식                          |
| Document Loading           | 다양한 문서 형식에서 텍스트와 메타데이터를 추출하는 과정                          |
| PDF Parsing                | PDF 내부 구조를 해석하여 텍스트와 메타데이터를 추출하는 과정                      |
| Text Preprocessing         | 노이즈를 제거하여 검색과 임베딩에 적합한 텍스트로 정제하는 과정                      |
| RAG Indexing               | 문서를 청킹·임베딩하여 벡터 DB에 저장하는 사전 준비 과정                        |
| Retrieval                  | 사용자 질문과 가장 유사한 문서 청크를 벡터 DB에서 검색하는 단계                    |
| Generation                 | 검색된 문서를 근거로 LLM이 최종 답변을 생성하는 단계                          |
| Context Window             | LLM이 한 번에 처리할 수 있는 최대 토큰 범위                              |
| Prompt Engineering for RAG | 검색된 문서를 근거로만 답변하도록 프롬프트를 설계하는 기법                         |
