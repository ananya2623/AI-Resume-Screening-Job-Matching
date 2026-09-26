"""
Skill extraction backed by a curated skills taxonomy (data/skills_db.json).
"""

import json
import os
import re

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "skills_db.json")


def load_skills_db(path: str = _DB_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_skills(db: dict) -> list:
    """Combine every skill-category list into one flat, de-duplicated list."""
    flat = []
    for key, value in db.items():
        if key == "education_keywords":
            continue
        flat.extend(value)
    return sorted(set(flat))


def extract_skills(text: str, db: dict = None) -> set:
    """Return the set of taxonomy skills mentioned anywhere in `text`."""
    if db is None:
        db = load_skills_db()
    vocab = _flatten_skills(db)
    text_lower = text.lower()

    found = set()
    for skill in vocab:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def highest_education_level(text: str, db: dict = None) -> tuple:
    """
    Return (level_name, level_score 0-4) for the highest education
    credential mentioned in `text`, based on db["education_keywords"].
    """
    if db is None:
        db = load_skills_db()
    edu_map = db.get("education_keywords", {})
    text_lower = text.lower()

    best_name, best_score = "none detected", 0
    for keyword, score in edu_map.items():
        if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower) and score > best_score:
            best_name, best_score = keyword, score
    return best_name, best_score
