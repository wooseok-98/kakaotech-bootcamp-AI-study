| 개념 | 한줄 정리 |
|--------|--------|
| LangChain | LLM, 데이터, 프롬프트, 체인을 연결하여 AI 애플리케이션을 개발하는 프레임워크 |
| Messages | 대화형 LLM에서 사용하는 메시지 객체(Human, AI, System 등) |
| LCEL (LangChain Expression Language) | 연산자(&#124;)로 체인을 선언적으로 연결하는 LangChain의 파이프라인 문법 |
| PromptTemplate | 변수 값을 삽입하여 동적 프롬프트를 생성하는 템플릿 |
| ChatPromptTemplate | 대화형 메시지 구조를 기반으로 프롬프트를 생성하는 템플릿 |
| FewShotPromptTemplate | 예제(Example)를 포함하여 모델이 원하는 형식을 학습하도록 만드는 프롬프트 템플릿 |
| MessagesPlaceholder | 이전 대화 기록이나 메시지 목록을 프롬프트에 동적으로 삽입하는 객체 |
| Structured Output | LLM의 응답을 JSON 등 정해진 구조로 출력하도록 만드는 기능 |
| JsonOutputParser | LLM 응답을 JSON 형태로 파싱하는 출력 파서 |
| PydanticOutputParser | Pydantic 모델을 기준으로 응답을 검증하고 객체로 변환하는 출력 파서 |
| Runnable Interface | LangChain 컴포넌트를 동일한 방식으로 실행할 수 있게 하는 공통 인터페이스 |
| RunnablePassthrough | 입력 데이터를 수정하지 않고 다음 단계로 그대로 전달하는 Runnable |
| RunnableParallel | 여러 Runnable을 동시에 실행하여 결과를 병렬로 반환하는 Runnable |
| RunnableBranch | 조건에 따라 서로 다른 Runnable을 선택하여 실행하는 Runnable |
| RunnableWithMessageHistory | 대화 기록을 자동으로 관리하며 체인을 실행하는 Runnable |
| Message Trimming | 토큰 제한을 고려해 오래된 대화 메시지를 제거하거나 요약하는 기능 |
| Document | 텍스트와 메타데이터를 함께 저장하는 LangChain의 문서 객체 |
| Document Loader | 외부 데이터 파일이나 웹페이지를 Document 객체로 변환하는 도구 |
| TextLoader | 텍스트(.txt) 파일을 불러와 Document로 변환하는 로더 |
| PyPDFLoader | PDF 파일을 읽어 페이지별 Document로 변환하는 로더 |
| WebBaseLoader | 웹페이지의 내용을 수집하여 Document로 변환하는 로더 |
| DirectoryLoader | 특정 디렉토리 내 여러 파일을 한 번에 불러오는 로더 |
| Text Splitter | 긴 문서를 검색·임베딩에 적합한 작은 청크로 나누는 도구 |
| RecursiveCharacterTextSplitter | 여러 구분자를 우선순위대로 시도해 의미 단위를 보존하며 분할하는 분할기 |
| CharacterTextSplitter | 단일 구분자 하나만 기준으로 분할하는 가장 단순한 분할기 |
| TokenTextSplitter | 토큰 수 기준으로 분할해 모델 토큰 한도에 정확히 맞추는 분할기 |
| Embeddings | 텍스트를 의미를 담은 고정 길이 실수 벡터로 변환하는 모델 |
| VectorStore Interface | 임베딩 벡터를 저장하고 유사도 기반으로 검색하는 저장소의 공통 인터페이스 |
| Chroma | 영구 저장이 가능한 오픈소스 로컬 벡터 DB |
| FAISS | 대용량·메모리 기반 고속 유사도 검색 라이브러리 |
| Retriever | 쿼리에 관련된 Document 리스트를 반환하는 검색 추상화 인터페이스 |
| Ensemble Retriever | 여러 Retriever의 결과를 가중치로 결합하는 하이브리드 검색 |
| Multi-Query Retriever | LLM으로 질문을 여러 변형으로 확장해 누락을 줄이는 검색 |
| Contextual Compression | 검색된 문서에서 질문과 무관한 부분을 압축·필터링하는 기법 |
| Parent Document Retriever | 작은 청크로 검색하고 응답에는 큰 부모 문서를 반환하는 검색 |
| LangSmith | LLM 앱을 추적·디버깅·평가·운영하는 통합 관측 플랫폼 |
| Tracing | 체인 실행 단계별 입출력·시간·토큰·비용을 트리로 기록하는 기능 |
| Dataset | 평가에 쓸 입력과 기대 출력 쌍의 모음 |
| Evaluator | 데이터셋에 대해 체인 응답 품질을 자동 채점하는 도구 |
| Prompt Management | 프롬프트를 코드와 분리해 버전 관리·공유·재사용하는 기능 |