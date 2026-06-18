# 데이터 입력 (Data Input)

## 개요

데이터 입력 단계는 다양한 원본 문서(PDF, HTML, Markdown, JSON 등)에서 텍스트를 추출하여 RAG 파이프라인에 투입할 수 있는 형태로 변환하는 과정입니다.

RAG 파이프라인:

`Document Loading → Text Preprocessing → Chunking → Embedding → Vector Store`

---

# 1. Document Loading

## 정의

Document Loading은 원본 문서에서 텍스트와 메타데이터를 추출하여 RAG 파이프라인이 처리할 수 있는 형태로 변환하는 과정입니다.

## 주요 문서 형식

| 형식 | 대표 도구 |
|--------|--------|
| TXT / Markdown | open() |
| PDF | PyMuPDF, pdfplumber |
| HTML | BeautifulSoup |
| JSON | json |
| CSV | pandas |
| DOCX | python-docx |

## 사용하는 이유

- 문서 형식마다 구조가 다름
- 텍스트 손실 최소화
- 메타데이터 수집 가능
- 검색 필터링 지원

## 반환 데이터 예시

```python
{
    "text": "...",
    "metadata": {
        "source": "manual.pdf",
        "page": 3
    }
}
```

## 기본 절차

1. 파일 형식 확인
2. 적절한 파서 선택
3. 텍스트 추출
4. 메타데이터 수집
5. 결과 반환

## 실무 포인트

- Markdown은 구조를 그대로 유지 가능
- HTML은 태그 제거 필요
- JSON은 필요한 필드만 추출
- 메타데이터는 Vector DB에 함께 저장

---

# 2. PDF Parsing

## 정의

PDF Parsing은 PDF 파일 내부 구조를 해석하여 텍스트, 표, 메타데이터를 추출하는 과정입니다.

## PDF가 어려운 이유

PDF는 단순 텍스트 파일이 아닙니다.

내부적으로는 문자와 위치 좌표가 저장되어 있기 때문에 별도의 PDF 파서가 필요합니다.

## PDF 유형

| 유형 | 특징 | 난이도 |
|--------|--------|--------|
| 텍스트 PDF | 문자 데이터 존재 | 낮음 |
| 스캔 PDF | 이미지 형태 | 높음 |
| 혼합 PDF | 텍스트 + 이미지 | 중간 |
| 다단 레이아웃 | 칼럼 구조 | 중간 |
| 표 포함 PDF | 표 구조 존재 | 높음 |

## 주요 라이브러리

| 라이브러리 | 특징 |
|-----------|------|
| PyPDF2 | 간단한 텍스트 추출 |
| pdfplumber | 표 추출 지원 |
| PyMuPDF | 빠르고 RAG 친화적 |
| Tesseract OCR | 스캔 PDF 처리 |

## 사용하는 이유

- PDF는 일반 read()로 읽을 수 없음
- 텍스트 추출 필요
- 표 데이터 활용 가능
- 페이지 정보 저장 가능
- 대량 PDF 자동 처리 가능

## PyMuPDF 기본 예제

```python
import pymupdf

doc = pymupdf.open("sample.pdf")

for page in doc:
    text = page.get_text()
    print(text)

doc.close()
```

## 페이지 메타데이터 예시

```python
{
    "source": "sample.pdf",
    "page": 2,
    "total_pages": 10
}
```

## RAG 활용 방법

1. PDF 열기
2. 페이지별 텍스트 추출
3. 메타데이터 저장
4. Chunking 수행
5. Embedding 생성
6. Vector DB 저장

## 실무 포인트

- 일반적으로 PyMuPDF 사용 권장
- 스캔 PDF는 OCR 필요
- 표가 중요하면 pdfplumber 고려
- pymupdf4llm 사용 시 Markdown 변환 가능

---

# 정리

| 기술 | 역할 |
|--------|--------|
| Document Loading | 다양한 문서 형식에서 텍스트 추출 |
| PDF Parsing | PDF 문서 전용 텍스트 추출 |
| 결과물 | 텍스트 + 메타데이터 |
| 다음 단계 | Chunking |
