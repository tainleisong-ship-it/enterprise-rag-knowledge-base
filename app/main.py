import logging
import sys
import threading
import time
import webbrowser
from pathlib import Path

# 确保 `python app/main.py` 直接运行时项目根目录在 sys.path 中
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import uvicorn
from fastapi import FastAPI, HTTPException

from app.api.schemas import HealthResponse, QueryRequest, QueryResponse, SourceDocument
from app.rag.chain import rag_chain
from app.services.cache import cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise RAG Knowledge Base",
    description="基于混合检索（BM25 + 向量）与规则重排序的企业级问答系统",
    version="1.0.0",
)


@app.get("/", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", docs="/docs", cache_available=cache.available)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    cache_key = cache.make_key(req.question)

    cached = cache.get(cache_key)
    if cached:
        logger.info("Cache hit: %s", cache_key)
        return QueryResponse(
            question=req.question,
            answer=cached["answer"],
            sources=[SourceDocument(**s) for s in cached["sources"]],
            cached=True,
        )

    try:
        result = rag_chain.run(req.question, top_k=req.top_k)
    except Exception as e:
        logger.exception("RAG chain failed")
        raise HTTPException(status_code=500, detail=f"问答链路异常: {e}")

    cache.set(cache_key, result)
    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        sources=[SourceDocument(**s) for s in result["sources"]],
        cached=False,
    )


def _open_browser() -> None:
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000/docs")


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
