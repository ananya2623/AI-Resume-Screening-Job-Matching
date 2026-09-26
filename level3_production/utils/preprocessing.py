"""
Text cleaning / preprocessing utilities, plus lightweight heuristics
for years-of-experience extraction used by the matching engine.
"""

import re
import string

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "as", "by", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "from", "will", "we", "you",
    "your", "our", "their", "i", "he", "she", "they", "them", "us", "have",
    "has", "had", "do", "does", "did", "not", "so", "than", "then", "there",
}


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation/digits/extra whitespace, drop stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def light_clean(text: str) -> str:
    """Milder cleaning that keeps numbers/punctuation — used before skill/year regex."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_years_of_experience(raw_text: str) -> float:
    """
    Best-effort extraction of years of experience mentioned in a resume,
    e.g. "5+ years of experience", "3 years", "over 4 years".
    Returns the maximum value found, or 0.0 if none found.
    """
    pattern = r"(\d+(?:\.\d+)?)\s*\+?\s*years?"
    matches = re.findall(pattern, raw_text.lower())
    years = [float(m) for m in matches if float(m) < 50]  # sanity filter
    return max(years) if years else 0.0


def extract_required_years(jd_text: str) -> float:
    """Same extraction, applied to a job description's requirement lines."""
    return extract_years_of_experience(jd_text)
