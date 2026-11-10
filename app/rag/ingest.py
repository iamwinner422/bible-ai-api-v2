"""Ingestion Version B - Hybrid (Dense + BM25)"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app import config
from app.rag.embeddings import encode_texts, load_embedding_model

logger = logging.getLogger(__name__)

def load_verses_from_json(json_path: Path) -> list[dict[str, Any]]:
    """Charge la Bible LSG selon la structure exacte de ton fichier."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    verses = []

    for testament in data.get("Testaments", []):
        for book in testament.get("Books", []):
            book_title = book.get("Text", "Inconnu").strip()

            for chapter in book.get("Chapters", []):
                chapter_id = chapter.get("ID")
                chapter_num = str(chapter_id) if chapter_id is not None else "1"

                for verse in chapter.get("Verses", []):
                    text = verse.get("Text", "").strip()
                    if not text:
                        continue

                    verse_id = verse.get("ID")
                    verse_num = str(verse_id) if verse_id is not None else "1"

                    verses.append({
                        "book": book_title,
                        "chapter": chapter_num,
                        "verse": verse_num,
                        "text": text,
                        "reference": f"{book_title} {chapter_num}:{verse_num}"
                    })

    return verses

def filter_verses(verses: list[dict]) -> list[dict]:
    """Filtrage plus strict pour améliorer la qualité."""
    return [
        v for v in verses
        if len(v["text"]) >= config.MIN_TEXT_LENGTH
        and len(v["text"].split()) >= config.MIN_WORD_COUNT
    ]

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Chargement du JSON LSG...")
    verses = load_verses_from_json(config.BIBLE_JSON_PATH)
    logger.info(f"Total versets bruts : {len(verses)}")

    filtered = filter_verses(verses)
    logger.info(f"Après filtrage strict : {len(filtered)}")

    texts = [v["text"] for v in filtered]
    tokenized = [t.lower().split() for t in texts]

    # === Dense index (FAISS) ===
    logger.info("Chargement du modèle d'embedding...")
    model = load_embedding_model()

    logger.info("Création des embeddings...")
    embeddings = encode_texts(model, texts)

    index = faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)
    index.add(embeddings)

    # === BM25 ===
    logger.info("Création de l'index BM25...")
    bm25 = BM25Okapi(tokenized)

    # Sauvegarde
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))

    with open(config.MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    with open(config.BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    logger.info("Ingestion terminée avec succès (Dense + BM25).")

if __name__ == "__main__":
    main()