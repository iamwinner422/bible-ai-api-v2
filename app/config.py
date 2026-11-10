from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

BIBLE_JSON_PATH = DATA_DIR / "bible_fr.json"
INDEX_PATH = DATA_DIR / "index.faiss"
MAPPING_PATH = DATA_DIR / "mapping.json"
BM25_PATH = DATA_DIR / "bm25.pkl"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384

FAISS_TOP_K = 40
RERANK_TOP_K = 8
MAX_QUERY_LENGTH = 300
MIN_TEXT_LENGTH = 25          # plus strict
MIN_WORD_COUNT = 5