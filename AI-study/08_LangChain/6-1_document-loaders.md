# 6-1. 문서 추상화와 로딩 (Document · Document Loader)

외부 데이터(텍스트, PDF, 웹페이지)를 LangChain의 표준 단위인 **Document** 로 불러오는 도구

> RAG 파이프라인의 첫 단계 — 다양한 소스를 일관된 형식으로 변환해야 이후 분할·임베딩이 가능

---

## Document

**텍스트(`page_content`)** 와 **메타데이터(`metadata`)** 를 함께 담는 LangChain의 문서 객체

```python
from langchain_core.documents import Document

doc = Document(
    page_content="LangChain은 LLM 애플리케이션 프레임워크다.",
    metadata={"source": "intro.txt", "page": 1},
)
print(doc.page_content)
print(doc.metadata["source"])
```

| 속성 | 설명 |
|---|---|
| `page_content` | 실제 텍스트 내용 |
| `metadata` | 출처, 페이지 번호, 작성일 등 부가 정보 (dict) |

> metadata는 검색 결과 필터링·출처 표시에 활용

---

## Document Loader

외부 파일·웹페이지를 **Document 객체 리스트로 변환**하는 도구

| 메서드 | 설명 |
|---|---|
| `load()` | 전체를 한 번에 읽어 `List[Document]` 반환 |
| `lazy_load()` | 제너레이터로 하나씩 읽음 (대용량 메모리 절약) |

---

## TextLoader

`.txt` 텍스트 파일을 읽어 Document로 변환

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("note.txt", encoding="utf-8")
docs = loader.load()
print(docs[0].page_content)
```

> 한글 파일은 `encoding="utf-8"` 명시 권장 (인코딩 오류 방지)

---

## PyPDFLoader

PDF 파일을 읽어 **페이지별 Document** 로 변환 (페이지 번호가 metadata에 저장됨)

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("paper.pdf")
docs = loader.load()
print(len(docs))                  # 페이지 수만큼 Document 생성
print(docs[0].metadata["page"])   # 0
```

> `pip install pypdf` 필요

---

## WebBaseLoader

웹페이지 URL의 내용을 수집해 Document로 변환

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com/article")
docs = loader.load()
print(docs[0].page_content[:200])
```

> `pip install beautifulsoup4` 필요 — HTML 태그를 제거하고 텍스트만 추출

---

## DirectoryLoader

특정 디렉토리 안의 **여러 파일을 한 번에** 로드 (패턴 매칭)

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    "./docs",
    glob="**/*.txt",            # 하위 폴더 포함 모든 .txt
    loader_cls=TextLoader,      # 각 파일에 적용할 로더
    loader_kwargs={"encoding": "utf-8"},
)
docs = loader.load()
```

| 매개변수 | 설명 |
|---|---|
| `glob` | 파일 매칭 패턴 (`**/*.pdf` 등) |
| `loader_cls` | 개별 파일 로딩에 쓸 로더 클래스 |
| `loader_kwargs` | 로더에 전달할 추가 인자 |

### 로더 요약

| 로더 | 대상 | 비고 |
|---|---|---|
| `TextLoader` | .txt | 인코딩 주의 |
| `PyPDFLoader` | .pdf | 페이지별 분리 |
| `WebBaseLoader` | 웹페이지 | HTML 파싱 |
| `DirectoryLoader` | 폴더 내 다수 파일 | glob 패턴 |
