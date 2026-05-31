import logging
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Request, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import chromadb
import pypdf
import io
import shutil
from fastapi import Form
import os
from dotenv import load_dotenv
import traceback
import tempfile
import requests as http_requests

# ── 로깅 설정 ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── 환경변수 로드 ────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY         = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL           = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
RAG_MODEL_NAME       = os.getenv("RAG_MODEL_NAME", "jhgan/ko-sbert-nli")
CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE", "300"))
OVERLAP              = int(os.getenv("CHUNK_OVERLAP", "50"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
_TEMP_BASE           = Path(tempfile.gettempdir()) / "rag_chatbot"
CHROMA_DB_PATH       = str(_TEMP_BASE / "chroma_data")
DATA_DIR             = _TEMP_BASE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# CORS 허용 출처 (.env의 ALLOWED_ORIGINS를 콤마로 구분. 미설정 시 전체 허용)
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

# ── 전역 상태 ────────────────────────────────────────────────
DB: list              = []
DB_EMBEDDINGS         = None
rag_model             = None
collection            = None


# ── Groq API 호출 (자동 재시도 포함) ─────────────────────────
import time

def call_llm(messages: list[dict], max_retries: int = 3) -> str:
    """Groq API 호출 (OpenAI 호환 형식). 429 시 자동 재시도."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY가 설정되지 않았습니다. servers/.env 파일을 확인하세요.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    for attempt in range(max_retries):
        try:
            resp = http_requests.post(url, json=body, headers=headers, timeout=60)

            if resp.status_code == 429:
                wait = min(15, 5 * (attempt + 1))
                log.warning("Groq 429 한도 초과 — %d초 대기 후 재시도 (%d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                error_body = resp.text[:500]
                log.error("Groq API 오류: HTTP %d\n%s", resp.status_code, error_body)
                try:
                    err_msg = resp.json().get("error", {}).get("message", error_body)
                except Exception:
                    err_msg = error_body

                if resp.status_code in (401, 403):
                    raise HTTPException(status_code=401, detail=f"Groq API 인증 실패: {err_msg}")
                raise HTTPException(status_code=502, detail=f"Groq API 오류 (HTTP {resp.status_code}): {err_msg}")

            data = resp.json()
            return data["choices"][0]["message"]["content"]

        except HTTPException:
            raise
        except Exception as e:
            log.error("Groq API 호출 실패: %s", e)
            if attempt == max_retries - 1:
                raise HTTPException(status_code=503, detail=f"Groq API 연결 실패: {str(e)}")
            time.sleep(3)

    raise HTTPException(status_code=429, detail="Groq API 요청 한도 초과. 잠시 후 다시 시도해주세요.")


# ── 텍스트 추출 ──────────────────────────────────────────────
def extract_text(file_path=None, content=None, file_ext=None) -> str:
    def _read_pdf(reader: pypdf.PdfReader) -> str:
        return "\n".join(p.extract_text() or "" for p in reader.pages)

    if file_path:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return _read_pdf(pypdf.PdfReader(file_path))
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")

    if content is not None and file_ext:
        if file_ext == ".pdf":
            return _read_pdf(pypdf.PdfReader(io.BytesIO(content)))
        return content.decode("utf-8", errors="ignore")

    return ""


# ── 고정 크기 청킹 (fallback) ────────────────────────────────
def chunk_text(text: str, size=CHUNK_SIZE, overlap=OVERLAP) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be less than chunk size")
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


# ── 조항 단위 청킹 ────────────────────────────────────────────
def chunk_by_article(text: str, max_size=300) -> list[str]:
    articles = re.split(r'(?=^## )', text, flags=re.MULTILINE)
    articles = [a.strip() for a in articles if len(a.strip()) > 20]
    chunks = []
    for article in articles:
        if len(article) <= max_size:
            chunks.append(article)
        else:
            lines   = article.split('\n')
            title   = lines[0]
            body    = '\n'.join(lines[1:])
            for sc in chunk_text(body, size=max_size, overlap=50):
                chunks.append(f"{title}\n{sc}")
    return chunks


# ── Semantic Chunking ────────────────────────────────────────
def semantic_chunk(
    text: str,
    model: SentenceTransformer,
    buffer_size: int  = 1,
    percentile: float = 85.0,
    max_chunk_size: int = 500
) -> list[str]:
    sentences = re.split(r'(?<=[.!?。])\s+|\n{2,}', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    if len(sentences) > 200:
        raise ValueError(f"Too many sentences ({len(sentences)})")

    embeddings = model.encode(sentences, batch_size=32, show_progress_bar=False)

    def _group_vec(start: int, end: int) -> np.ndarray:
        vecs = embeddings[max(0, start):min(len(embeddings), end)]
        mean = np.mean(vecs, axis=0)
        return mean / (np.linalg.norm(mean) + 1e-12)

    distances = []
    for i in range(len(sentences) - 1):
        a = _group_vec(i - buffer_size, i + buffer_size + 1)
        b = _group_vec(i + 1 - buffer_size, i + 1 + buffer_size + 1)
        distances.append(1.0 - float(np.dot(a, b)))

    threshold = float(np.percentile(distances, percentile))
    chunks, current = [], [sentences[0]]

    for i, dist in enumerate(distances):
        if dist > threshold:
            chunks.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    if current:
        chunks.append(" ".join(current))

    final = []
    for c in chunks:
        if len(c) > max_chunk_size:
            final.extend(chunk_text(c, size=max_chunk_size, overlap=50))
        else:
            final.append(c)

    return [c for c in final if c.strip()]


# ── 벡터 정규화 ──────────────────────────────────────────────
def normalize(vecs: np.ndarray) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim == 1:
        return arr / (np.linalg.norm(arr) + 1e-12)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)


# ── 파일 확장자에 따른 최적 청킹 전략 선택 ─────────────────────
def smart_chunk(text: str, file_ext: str, model: SentenceTransformer) -> list[str]:
    if file_ext == ".md":
        try:
            chunks = chunk_by_article(text, max_size=300)
            if chunks:
                log.info("MD 조항 기반 청킹 완료: %d개 청크", len(chunks))
                return chunks
        except Exception as e:
            log.warning("MD 조항 기반 청킹 실패: %s", e)

    try:
        chunks = semantic_chunk(text, model)
        if chunks:
            log.info("Semantic Chunking 완료: %d개 청크", len(chunks))
            return chunks
    except Exception as e:
        log.warning("Semantic Chunking 실패: %s", e)

    if file_ext != ".md":
        try:
            chunks = chunk_by_article(text, max_size=300)
            if chunks:
                log.info("조항 기반 청킹(fallback) 완료: %d개 청크", len(chunks))
                return chunks
        except Exception as e:
            log.warning("조항 기반 청킹 실패: %s", e)

    chunks = chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP)
    log.info("고정 크기 청킹(최종 fallback) 완료: %d개 청크", len(chunks))
    return chunks


# ── DB 추가 ──────────────────────────────────────────────────
def add_to_db(chunks: list[str], source: str) -> int:
    global DB, DB_EMBEDDINGS
    start_id  = len(DB)
    new_items = [{"id": str(start_id + i), "text": c, "source": source} for i, c in enumerate(chunks)]
    DB.extend(new_items)

    new_embs = normalize(rag_model.encode([item["text"] for item in new_items], batch_size=32, show_progress_bar=False))
    DB_EMBEDDINGS = new_embs if DB_EMBEDDINGS is None else np.vstack([DB_EMBEDDINGS, new_embs])

    collection.add(
        documents =[item["text"]              for item in new_items],
        embeddings=new_embs.tolist(),
        metadatas =[{"source": item["source"]} for item in new_items],
        ids       =[item["id"]                 for item in new_items],
    )
    return len(new_items)


def reset_chroma_db():
    global DB, DB_EMBEDDINGS
    DB, DB_EMBEDDINGS = [], None
    chroma_path = Path(CHROMA_DB_PATH)
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)


# ── 앱 생명주기 ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_model, collection

    log.info("AI 모델 초기화 중...")
    if not GROQ_API_KEY:
        log.warning(".env 파일에 GROQ_API_KEY가 설정되지 않았습니다.")
    log.info("임베딩 모델 로딩: %s", RAG_MODEL_NAME)
    rag_model = SentenceTransformer(RAG_MODEL_NAME)

    reset_chroma_db()
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(
        "rag_docs", metadata={"hnsw:space": "cosine"}
    )

    log.info("서버 준비 완료 (임베딩: %s / LLM: Groq %s)", RAG_MODEL_NAME, GROQ_MODEL)
    yield
    log.info("서버 종료")


# ── FastAPI 앱 ───────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins    =ALLOWED_ORIGINS,
    allow_methods    =["*"],
    allow_headers    =["*"],
    allow_credentials=False,
)


# ── 요청 모델 ────────────────────────────────────────────────
class ChatReq(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message는 비어있을 수 없습니다.")
        return v.strip()


class JsonReq(BaseModel):
    message: str


# ── 엔드포인트 ───────────────────────────────────────────────

@app.post("/simpleparam")
def simple_param(message: str = Form(...)):
    log.info("simple_param: %s", message)
    return {
        "type": "simple_request",
        "message": f"받은 메시지: {message}",
        "preflight": "불필요 (application/x-www-form-urlencoded)"
    }


@app.post("/simplejson")
def simple_json(req: JsonReq):
    log.info("simple_json: %s", req.message)
    return {
        "type": "preflight_request",
        "message": f"받은 메시지: {req.message}",
        "preflight": "필요 (application/json)"
    }


@app.post("/chat")
def chat(req: ChatReq):
    answer = call_llm([{"role": "user", "content": req.message}])
    return {"answer": answer}


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    log.info("=" * 50)
    log.info("업로드 요청 수신: filename=%s, content_type=%s", file.filename, file.content_type)

    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다.")

    allowed_extensions = {".txt", ".md", ".pdf"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. ({', '.join(allowed_extensions)} 만 허용)"
        )

    try:
        content = await file.read()
        log.info("파일 크기: %d bytes", len(content))

        if len(content) == 0:
            raise HTTPException(status_code=422, detail="빈 파일입니다.")

        text = extract_text(content=content, file_ext=file_ext)
        log.info("텍스트 추출 완료: %d자", len(text))

        if not text.strip():
            raise HTTPException(status_code=422, detail="파일에 읽을 수 있는 내용이 없습니다.")

        chunks = smart_chunk(text, file_ext, rag_model)

        if not chunks:
            raise HTTPException(status_code=422, detail="청킹 결과가 없습니다. 파일 내용을 확인해주세요.")

        log.info("청킹 완료: %d개 청크, DB 저장 시작...", len(chunks))
        chunks_added = add_to_db(chunks, file.filename)

        safe_filename = file.filename.encode("utf-8", errors="replace").decode("utf-8")
        save_path = DATA_DIR / safe_filename
        save_path.write_bytes(content)

        log.info("업로드 완료: %s (%d 청크)", file.filename, chunks_added)
        log.info("=" * 50)

        return {
            "success": True,
            "message": f"'{file.filename}' 업로드 완료! ({chunks_added}개 청크 추가)",
            "chunks_added": chunks_added,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류가 발생했습니다: {str(e)}")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": RAG_MODEL_NAME,
        "llm": f"Groq {GROQ_MODEL}",
        "db_size": len(DB),
        "collection_count": collection.count() if collection else 0,
    }


@app.post("/search")
def search(
    req:   ChatReq,
    top_k: int  = Query(5, ge=1, le=20),
    debug: bool = Query(True),
):
    if DB_EMBEDDINGS is None or len(DB) == 0:
        raise HTTPException(status_code=404, detail="DB가 비어있습니다. 먼저 문서를 업로드해주세요.")

    query_vec = normalize(rag_model.encode(req.message))
    scores    = np.dot(DB_EMBEDDINGS, query_vec)
    idxs      = np.argsort(scores)[::-1][:top_k]

    candidates = [{
        "rank":      r + 1,
        "score":     float(scores[i]),
        "source":    DB[i]["source"],
        "full_text": DB[i]["text"],
    } for r, i in enumerate(idxs)]

    best     = candidates[0]
    is_valid = best["score"] >= SIMILARITY_THRESHOLD

    return {
        "answer":     best["full_text"] if is_valid else "관련된 내용을 찾을 수 없습니다.",
        "score":      best["score"],
        "source":     best["source"] if is_valid else None,
        "candidates": candidates if debug else None,
    }


@app.post("/integrated-chat")
def integrated_chat(
    req:       ChatReq,
    n_results: int  = Query(3, ge=1, le=8),
    debug:     bool = Query(False),
):
    if collection is None:
        raise HTTPException(status_code=503, detail="벡터 DB가 초기화되지 않았습니다.")

    q       = normalize(rag_model.encode([req.message]))
    results = collection.query(query_embeddings=q.tolist(), n_results=n_results)

    docs  = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    candidates = []
    for doc, meta, dist in zip(docs, metas, dists):
        sim = 1.0 - float(dist)
        candidates.append({"source": meta.get("source"), "similarity": sim, "text": doc})

    picked = sorted(
        [c for c in candidates if c["similarity"] >= SIMILARITY_THRESHOLD],
        key=lambda x: x["similarity"], reverse=True
    )[:n_results]

    if picked:
        context    = "\n\n".join(
            f"[출처: {p['source']} | 유사도: {p['similarity']:.2f}]\n{p['text']}" for p in picked
        )
        source_tag = ", ".join(sorted({p["source"] for p in picked}))
    else:
        context    = "(검색결과 없음)"
        source_tag = "LLM 일반 지식"

    prompt = f"""
너는 회사 내부 규정안내를 도와주는 사내 문서 도우미야.

[참고 문서]의 내용을 바탕으로 사용자의 질문에 답변해. 문서에 없는 내용은 지어내지 말고 모르거나 없다고 대답해.
문서를 기반으로 답변을 생성할 때 그 답변과 관련된 문서의 내용을 직접 확인해서 사용자가 검증해볼 수 있도록 내용의 위치를 같이 출력해.

출력방식은 간결하고 보기 좋게 깔끔하게 정리해서 출력해.
문서의 검증 위치는 아래 형식으로 작성해.
" [문서명] - [행번호] "

너가 생성한 답변이 문서 내용과 위배되거나 없는 내용을 임의로 지어내었는지 반복적으로 재검증하고 틀린 부분을 수정해서 최종 답변해줘.
[참고문서]
{context}
""".strip()

    answer = call_llm([
        {"role": "system", "content": prompt},
        {"role": "user",   "content": req.message},
    ])

    payload: dict = {"answer": answer, "source": source_tag}
    if debug:
        payload["candidates"] = [{"source": c["source"], "similarity": round(c["similarity"], 4)} for c in candidates]
        payload["picked"]     = [{"source": p["source"], "similarity": round(p["similarity"], 4)} for p in picked]
    return payload


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
    )
<<<<<<< HEAD
=======
":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
    )
>>>>>>> ec00517230cf8ba8b447222d95f410f5e938ea83
