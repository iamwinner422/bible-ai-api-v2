import json
import pickle
from collections import defaultdict
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app import config
from app.rag.embeddings import load_cross_encoder, load_embedding_model
from app.rag.query_understanding import extract_important_terms, generate_queries

def normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-raw_scores))

def reciprocal_rank_fusion(results_list: list[list[dict]], k: int = 60) -> list[dict]:
    scores = defaultdict(float)
    docs = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            ref = doc["reference"]
            scores[ref] += 1.0 / (k + rank + 1)
            docs[ref] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused = []
    for ref, rrf_score in ranked:
        doc = docs[ref].copy()
        doc["rrf_score"] = rrf_score
        fused.append(doc)

    return fused

def load_pipeline():
    index = faiss.read_index(str(config.INDEX_PATH))
    with open(config.MAPPING_PATH, encoding="utf-8") as f:
        mapping = json.load(f)
    with open(config.BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)

    embed_model = load_embedding_model()
    cross_encoder = load_cross_encoder()
    return index, mapping, bm25, embed_model, cross_encoder

def _search_single_query(
    query: str,
    index: faiss.Index,
    mapping: list[dict],
    bm25: BM25Okapi,
    embed_model: SentenceTransformer,
    cross_encoder: CrossEncoder,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    query_clean = query.replace("\n", " ").strip()

    # Dense
    query_emb = embed_model.encode(
        [query_clean],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    _, dense_indices = index.search(query_emb, config.FAISS_TOP_K)

    # BM25
    tokenized_query = query_clean.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:config.FAISS_TOP_K]

    candidate_indices = list(set(dense_indices[0].tolist() + bm25_indices.tolist()))
    candidate_indices = [int(i) for i in candidate_indices if 0 <= i < len(mapping)]

    if not candidate_indices:
        return []

    candidates = []
    for idx in candidate_indices:
        entry = mapping[idx]
        text_clean = entry["text"].replace("\n", " ")
        candidates.append((idx, text_clean, entry))

    pairs = [[query_clean, text] for _, text, _ in candidates]
    raw_scores = cross_encoder.predict(pairs)
    scores = normalize_scores(np.asarray(raw_scores, dtype=np.float32))

    results = []
    for i, (_, _, entry) in enumerate(candidates):
        results.append({
            "reference": entry["reference"],
            "book": entry["book"],
            "chapter": entry["chapter"],
            "verse": entry["verse"],
            "text": entry["text"],
            "score": float(scores[i]),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def search(
    query: str,
    index: faiss.Index,
    mapping: list[dict],
    bm25: BM25Okapi,
    embed_model: SentenceTransformer,
    cross_encoder: CrossEncoder,
    top_k: int = config.RERANK_TOP_K,
    min_score: float = 0.10,
) -> list[dict[str, Any]]:
    """
    Recherche avancée avec Pseudo-Relevance Feedback (sans liste de thèmes).
    """

    # === Étape 1 : Première recherche (requête originale) ===
    first_results = _search_single_query(
        query, index, mapping, bm25, embed_model, cross_encoder, top_k=12
    )

    # === Étape 2 : Extraction de termes importants (Feedback) ===
    feedback_texts = [r["text"] for r in first_results[:6]]
    important_terms = extract_important_terms(feedback_texts, top_n=7)

    # === Étape 3 : Génération de plusieurs requêtes enrichies ===
    queries = generate_queries(query, feedback_terms=important_terms)

    # === Étape 4 : Recherche pour chaque reformulation ===
    all_results = []
    for q in queries:
        res = _search_single_query(
            q, index, mapping, bm25, embed_model, cross_encoder, top_k=20
        )
        all_results.append(res)

    # === Étape 5 : Fusion RRF ===
    fused = reciprocal_rank_fusion(all_results)

    # === Étape 6 : Filtrage final ===
    final = []
    seen_refs = set()
    for doc in fused:
        if doc["reference"] in seen_refs:
            continue
        if doc.get("score", 0) >= min_score or doc.get("rrf_score", 0) > 0.015:
            final.append(doc)
            seen_refs.add(doc["reference"])

    return final[:top_k]