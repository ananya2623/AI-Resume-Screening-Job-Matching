# AI-Powered Resume Screening & Job Matching System

A recruiter uploads a candidate **resume (PDF)** and a **job description**, and the
system returns a structured match report:

```
Match Score: 87%
Technical Skills: 92%
Experience Relevance: 81%
Education: 95%
Missing Skills: Docker, AWS
Recommended: Shortlist
```

The project is built in **three levels of increasing sophistication**, so you can
show progression from classic ML → modern NLP → a production-style app.

```
resume_matcher/
├── requirements.txt
├── README.md
├── sample_data/
│   ├── sample_job_description.txt
│   └── sample_resume.txt
├── level1_basic_ml/
│   └── app_level1.py         # TF-IDF + Cosine Similarity
├── level2_nlp/
│   └── app_level2.py         # Sentence Transformers + Skill Gap Analysis
└── level3_production/
    ├── app.py                # Full production-style Streamlit app
    ├── data/
    │   └── skills_db.json    # Curated skills taxonomy
    └── utils/
        ├── pdf_parser.py
        ├── preprocessing.py
        ├── skill_extractor.py
        ├── matching_engine.py
        └── visualization.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt
```

---

## Level 1 — Basic ML

```
Resume PDF → Text Extraction → Cleaning & Preprocessing → TF-IDF
    → Cosine Similarity → Match Score
```

Run it:
```bash
streamlit run level1_basic_ml/app_level1.py
```

What it demonstrates: classic bag-of-words retrieval (`TfidfVectorizer`) and
`cosine_similarity` from scikit-learn. This is the baseline every recruiting-tech
take-home should start with — fast, explainable, no GPU needed.

---

## Level 2 — NLP (Semantic Embeddings)

```
Resume + Job Description → Sentence Transformer → Semantic Embeddings
    → Similarity Calculation → Skill Gap Analysis
```

Run it:
```bash
streamlit run level2_nlp/app_level2.py
```

What it demonstrates: `sentence-transformers` (`all-MiniLM-L6-v2`) to embed the
resume and JD into dense vectors, so the match captures *meaning* rather than
exact keyword overlap (e.g. "led a team of engineers" ≈ "management experience").
Adds a skill-gap module that surfaces which required skills are missing from the
resume.

---

## Level 3 — Production-style Application

```
Resume PDF
   │
   ▼
Text Extraction
   │
   ▼
NLP Preprocessing
   │
 ┌─┴─┐
 ▼   ▼
Skill  Embeddings
 NLP
 └─┬─┘
   ▼
Matching Engine
   │
 ┌─┴──┐
 ▼    ▼
Score Gaps
 └─┬──┘
   ▼
Streamlit UI
```

Run it:
```bash
streamlit run level3_production/app.py
```

What it demonstrates: a modular codebase (`utils/`) with a `MatchingEngine` that
blends TF-IDF, semantic embeddings, a skills taxonomy (`data/skills_db.json`),
experience-relevance heuristics, and education matching into one weighted
**Match Score**, plus a **Skill Gap** breakdown, a recruiter recommendation
(Shortlist / Consider / Reject), and Matplotlib/Seaborn charts — all in a
polished Streamlit UI.

### Scoring model (Level 3)

| Component            | Weight | Method                                              |
|-----------------------|:------:|------------------------------------------------------|
| Technical Skills      | 40%    | Skills-taxonomy overlap (resume ∩ JD required skills) |
| Semantic Similarity   | 30%    | Sentence-transformer cosine similarity                |
| Experience Relevance  | 20%    | Years-of-experience + role-keyword heuristics         |
| Education Match       | 10%    | Degree-level keyword matching                         |

Overall recommendation thresholds: **≥75% → Shortlist**, **50–74% → Consider**,
**<50% → Reject**. Thresholds are configurable at the top of `matching_engine.py`.

---

## Tech stack

Python · Pandas · NumPy · Scikit-learn · NLP · TF-IDF / Sentence Transformers ·
Cosine Similarity · Streamlit · PyPDF2 / pdfplumber · Matplotlib / Seaborn · Git/GitHub
