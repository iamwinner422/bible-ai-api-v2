"""Compréhension de requête générale (sans liste de thèmes)"""

import re
from collections import Counter
from typing import List

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_important_terms(texts: List[str], top_n: int = 8) -> List[str]:
    """
    Extrait les termes les plus importants à partir des meilleurs versets trouvés.
    On filtre les mots trop courants.
    """
    stopwords = {
        "le", "la", "les", "un", "une", "des", "de", "du", "et", "en", "à", "au", "aux",
        "ce", "cet", "cette", "ces", "il", "elle", "ils", "elles", "nous", "vous", "je", "tu",
        "que", "qui", "quoi", "dont", "où", "ne", "pas", "plus", "moins", "très", "bien",
        "dans", "par", "pour", "sur", "avec", "sans", "sous", "comme", "mais", "ou", "donc",
        "car", "si", "se", "son", "sa", "ses", "mon", "ma", "mes", "ton", "ta", "tes",
        "leur", "leurs", "y", "a", "est", "sont", "être", "avoir", "fait", "faire",
        "dieu", "éternel", "seigneur", "jesus", "christ"  # trop fréquents dans la Bible
    }

    words = []
    for text in texts:
        cleaned = clean_text(text)
        tokens = [w for w in cleaned.split() if len(w) > 3 and w not in stopwords]
        words.extend(tokens)

    counter = Counter(words)
    return [word for word, _ in counter.most_common(top_n)]

def generate_queries(user_query: str, feedback_terms: List[str] | None = None) -> List[str]:
    """
    Génère plusieurs reformulations génériques de la requête.
    """
    queries = [
        user_query,
        f"versets sur {user_query}",
        f"que dit la bible sur {user_query}",
        f"verset encouragement {user_query}",
        f"parole de dieu concernant {user_query}",
    ]

    # Si on a des termes extraits des premiers résultats, on les utilise
    if feedback_terms:
        feedback_str = " ".join(feedback_terms)
        queries.append(f"{user_query} {feedback_str}")
        queries.append(feedback_str)

    # Déduplication
    seen = set()
    unique = []
    for q in queries:
        q_clean = q.strip().lower()
        if q_clean and q_clean not in seen:
            seen.add(q_clean)
            unique.append(q)

    return unique