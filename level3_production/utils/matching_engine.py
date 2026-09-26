"""
MatchingEngine
==============
Combines multiple signals into one weighted Match Score:

    Technical Skills (40%) : skills-taxonomy overlap (resume vs JD)
    Semantic Similarity (30%): sentence-transformer cosine similarity
    Experience Relevance (20%): years-of-experience heuristic
    Education Match (10%)  : degree-level keyword matching

Also produces a recruiter recommendation (Shortlist / Consider / Reject).
"""

from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

from . import preprocessing
from . import skill_extractor

# ---- configurable thresholds -------------------------------------------
WEIGHTS = {
    "technical_skills": 0.40,
    "experience": 0.20,
    "education": 0.10,
    "semantic": 0.30,
}
SHORTLIST_THRESHOLD = 75
CONSIDER_THRESHOLD = 50


@dataclass
class MatchResult:
    match_score: float
    technical_skills_score: float
    experience_score: float
    education_score: float
    semantic_score: float
    matched_skills: set
    missing_skills: set
    recommendation: str
    candidate_years: float = 0.0
    required_years: float = 0.0
    candidate_education: str = ""
    required_education: str = ""
    extra: dict = field(default_factory=dict)


class MatchingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None  # lazy-loaded, see .model property
        self._skills_db = skill_extractor.load_skills_db()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # ------------------------------------------------------------------
    def analyze(self, resume_text: str, jd_text: str) -> MatchResult:
        # --- Technical skills -----------------------------------------
        resume_skills = skill_extractor.extract_skills(resume_text, self._skills_db)
        jd_skills = skill_extractor.extract_skills(jd_text, self._skills_db)
        matched = resume_skills & jd_skills
        missing = jd_skills - resume_skills
        technical_skills_score = (
            round(len(matched) / len(jd_skills) * 100, 2) if jd_skills else 100.0
        )

        # --- Semantic similarity ----------------------------------------
        semantic_score = self._semantic_similarity(resume_text, jd_text)

        # --- Experience relevance ---------------------------------------
        candidate_years = preprocessing.extract_years_of_experience(resume_text)
        required_years = preprocessing.extract_required_years(jd_text)
        experience_score = self._experience_score(candidate_years, required_years)

        # --- Education -----------------------------------------------
        cand_edu_name, cand_edu_level = skill_extractor.highest_education_level(
            resume_text, self._skills_db
        )
        req_edu_name, req_edu_level = skill_extractor.highest_education_level(
            jd_text, self._skills_db
        )
        education_score = self._education_score(cand_edu_level, req_edu_level)

        # --- Weighted overall score --------------------------------------
        overall = (
            technical_skills_score * WEIGHTS["technical_skills"]
            + semantic_score * WEIGHTS["semantic"]
            + experience_score * WEIGHTS["experience"]
            + education_score * WEIGHTS["education"]
        )
        overall = round(overall, 2)

        recommendation = self._recommend(overall)

        return MatchResult(
            match_score=overall,
            technical_skills_score=technical_skills_score,
            experience_score=experience_score,
            education_score=education_score,
            semantic_score=semantic_score,
            matched_skills=matched,
            missing_skills=missing,
            recommendation=recommendation,
            candidate_years=candidate_years,
            required_years=required_years,
            candidate_education=cand_edu_name,
            required_education=req_edu_name,
        )

    # ------------------------------------------------------------------
    def _semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        emb_a = self.model.encode(resume_text, convert_to_tensor=True)
        emb_b = self.model.encode(jd_text, convert_to_tensor=True)
        score = util.cos_sim(emb_a, emb_b).item()
        return round(max(score, 0.0) * 100, 2)

    @staticmethod
    def _tfidf_similarity(resume_text: str, jd_text: str) -> float:
        """Kept for reference / offline fallback when embeddings aren't available."""
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(
            [preprocessing.clean_text(resume_text), preprocessing.clean_text(jd_text)]
        )
        score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return round(score * 100, 2)

    @staticmethod
    def _experience_score(candidate_years: float, required_years: float) -> float:
        if required_years <= 0:
            return 100.0 if candidate_years > 0 else 60.0  # no explicit requirement found
        if candidate_years >= required_years:
            return 100.0
        return round((candidate_years / required_years) * 100, 2)

    @staticmethod
    def _education_score(candidate_level: int, required_level: int) -> float:
        if required_level == 0:
            return 100.0 if candidate_level > 0 else 70.0
        if candidate_level >= required_level:
            return 100.0
        if candidate_level == 0:
            return 40.0
        return round((candidate_level / required_level) * 100, 2)

    @staticmethod
    def _recommend(score: float) -> str:
        if score >= SHORTLIST_THRESHOLD:
            return "Shortlist"
        if score >= CONSIDER_THRESHOLD:
            return "Consider"
        return "Reject"
