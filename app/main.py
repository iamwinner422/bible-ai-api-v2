from contextlib import asynccontextmanager
from typing import Any
import threading

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag.query_understanding import generate_queries
from app.rag.retrieve import load_pipeline, search

pipeline: dict[str, Any] = {}
pipeline_ready = threading.Event()

class VerseResult(BaseModel):
    reference: str
    book: str
    chapter: str
    verse: str
    text: str
    score: float

class SearchResponse(BaseModel):
    query: str
    expanded_query: str | None = None
    results: list[VerseResult]

def load_pipeline_background():
    try:
        index, mapping, bm25, embed_model, cross_encoder = load_pipeline()
        pipeline.update({
            "index": index,
            "mapping": mapping,
            "bm25": bm25,
            "embed_model": embed_model,
            "cross_encoder": cross_encoder,
        })
        pipeline_ready.set()
        print("✅ Pipeline hybrid prêt")
    except Exception as e:
        print(f"❌ Erreur chargement pipeline : {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=load_pipeline_background, daemon=True)
    thread.start()
    yield

app = FastAPI(
    title="Bible LSG Semantic Search API (Hybrid)",
    description="Recherche sémantique hybride dans la Bible Louis Segond. Répond uniquement par des versets.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    if not pipeline_ready.is_set():
        return {"status": "loading"}
    return {"status": "ok"}

@app.get("/search", response_model=SearchResponse)
def search_verses(
    q: str = Query(..., min_length=2, max_length=300),
    top_k: int = Query(7, ge=1, le=15),
):
    if not pipeline_ready.is_set():
        raise HTTPException(status_code=503, detail="Modèle en cours de chargement")

    generated_queries = generate_queries(q)
    expanded = generated_queries[1] if len(generated_queries) > 1 else q

    results = search(
        query=q,
        index=pipeline["index"],
        mapping=pipeline["mapping"],
        bm25=pipeline["bm25"],
        embed_model=pipeline["embed_model"],
        cross_encoder=pipeline["cross_encoder"],
        top_k=top_k,
    )

    return {
        "query": q,
        "expanded_query": expanded if expanded != q.lower() else None,
        "results": results,
    }