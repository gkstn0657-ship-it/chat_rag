# 사내 문서 RAG 챗봇

사내 문서 기반으로 질문에 답변하는 RAG(Retrieval-Augmented Generation) 기반 AI 챗봇입니다.

## 실습 목표

JavaScript와 Python 백엔드를 활용하여 단계별로 AI 챗봇을 고도화해봅니다.

## 프로젝트 구조

```
├── index.html          # 프론트엔드 메인 페이지
├── index_v2.html       # Step 2용 HTML (폼 기반 업로드)
├── style.css           # 스타일시트
│
├── script.js           # [기본] GPT 직접 통신 (console.log)
├── script_v1.js        # [Step 1] 채팅 UI 구현 (DOM 조작)
├── script_v2.js        # [Step 2] 업로드 기능 추가
├── script_v3.js        # [Step 3] 벡터 검색 (/search)
├── script_v4.js        # [Step 4] RAG 통합 채팅 (/integrated-chat)
│
└── servers/
    ├── main.py         # FastAPI 백엔드 서버
    ├── requirements.txt
    └── data/           # 문서 저장 폴더
```

> **참고**: `script_v1`~`script_v4`는 챗봇을 단계적으로 고도화한 학습 기록입니다.
> 현재 `index.html`은 최종본인 `script_v4.js`만 로드하며, 이전 버전들은 발전 과정을
> 보여주기 위해 의도적으로 보존하고 있습니다.

## 단계별 학습 가이드

### script.js (기본 코드)
- **학습 포인트**: fetch API, async/await, JSON 통신
- **기능**: `/chat` 엔드포인트로 GPT와 통신, 결과를 `console.log`로 확인
- **실행**: F12 개발자 도구에서 Console 탭 확인

### script_v1.js (Step 1)
- **학습 포인트**: DOM 조작, `document.createElement()`
- **변경점**: `console.log` → 화면에 채팅 말풍선으로 표시
- **핵심 함수**: `addMessage(type, text)` - div 태그를 동적으로 생성

### script_v2.js (Step 2)
- **학습 포인트**: HTML form, multipart/form-data
- **변경점**: 파일 업로드 기능 추가 (JS 코드 변경 최소화)
- **참고**: `index_v2.html` 사용 필요 (form 태그로 업로드 처리)

### script_v3.js (Step 3)
- **학습 포인트**: 벡터 검색 API 활용
- **변경점**: `/chat` → `/search` 엔드포인트 변경 (단 한 줄!)
- **결과**: 유사도 기반 문서 검색 결과 표시

### script_v4.js (Step 4)
- **학습 포인트**: RAG (Retrieval-Augmented Generation)
- **변경점**: `/search` → `/integrated-chat` 엔드포인트 변경
- **결과**: 검색 + GPT 생성이 결합된 최종 AI 챗봇

## 설치 및 실행

### 1. 의존성 설치

```bash
cd servers
pip install -r requirements.txt
```

### 2. 환경 설정

`servers/main.py`에서 OpenAI API 키 설정:

```python
OPENAI_API_KEY = "sk-..."  # 본인의 API 키 입력
GPT_MODEL = "gpt-4o-mini"  # 사용할 모델명
```

### 3. 서버 실행

```bash
cd servers
python main.py
```

서버가 `http://localhost:8000`에서 실행됩니다.

### 4. 프론트엔드 실행

`index.html`을 브라우저에서 열거나, Live Server 등으로 실행합니다.

사용할 스크립트 버전에 맞게 `index.html`의 script 태그를 수정하세요:

```html
<!-- 기본 버전 -->
<script src="script.js"></script>

<!-- Step 1: 채팅 UI -->
<script src="script_v1.js"></script>

<!-- Step 2: 업로드 (index_v2.html 사용 권장) -->
<script src="script_v2.js"></script>

<!-- Step 3: 벡터 검색 -->
<script src="script_v3.js"></script>

<!-- Step 4: RAG 통합 -->
<script src="script_v4.js"></script>
```

## API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/chat` | POST | GPT 직접 채팅 (RAG 없음) |
| `/search` | POST | 벡터 유사도 검색 |
| `/integrated-chat` | POST | RAG 통합 채팅 (검색 + 생성) |
| `/upload` | POST | 문서 업로드 (Referer 있으면 리다이렉트) |

### 요청 형식

```json
{ "message": "질문 내용" }
```

## 기술 스택

- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Backend**: FastAPI, Python
- **AI/ML**: OpenAI GPT, Sentence Transformers
- **Vector DB**: ChromaDB
- **PDF 처리**: pypdf
