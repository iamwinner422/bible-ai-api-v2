from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from app import config

def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)

def load_cross_encoder() -> CrossEncoder:
    return CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

def encode_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    cleaned = [t.replace("\n", " ").strip() for t in texts]
    return model.encode(
        cleaned,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )