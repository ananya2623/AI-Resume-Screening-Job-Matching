"""
Level 2 — NLP (Semantic Embeddings)
====================================
Pipeline:
    Resume + Job Description -> Sentence Transformer -> Semantic Embeddings
                              -> Similarity Calculation -> Skill Gap Analysis

Run:
    streamlit run level2_nlp/app_level2.py
"""

import re

import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util

# --------------------------------------------------------------------------
# A small but broad technical-skills vocabulary for gap analysis.
# (Level 3 replaces this with a richer, file-backed taxonomy.)
# --------------------------------------------------------------------------
SKILLS_VOCAB = [
    "python", "java", "c++", "sql", "r", "scala",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "nlp", "machine learning", "deep learning", "computer vision",
    "tf-idf", "sentence transformers", "transformers", "bert",
    "docker", "kubernetes", "aws", "azure", "gcp", "ci/cd",
    "git", "github", "linux", "rest api", "flask", "streamlit", "django",
    "matplotlib", "seaborn", "tableau", "power bi",
    "spark", "hadoop", "airflow", "mlops",
]


@st.cache_resource(show_spinner=False)
def load_model():
    """Load and cache the sentence-transformer model once per session."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(uploaded_file) -> str:
    text_chunks = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


def find_skills(text: str) -> set:
    """Return the set of vocabulary skills mentioned in `text`."""
    text_lower = text.lower()
    found = set()
    for skill in SKILLS_VOCAB:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def semantic_similarity(model, text_a: str, text_b: str) -> float:
    """Embed both texts and return cosine similarity as a percentage."""
    emb_a = model.encode(text_a, convert_to_tensor=True)
    emb_b = model.encode(text_b, convert_to_tensor=True)
    score = util.cos_sim(emb_a, emb_b).item()
    return round(score * 100, 2)


def main():
    st.set_page_config(page_title="Resume Matcher — Level 2 (NLP)", page_icon="🧠")
    st.title("🧠 Resume ↔ Job Match — Level 2: Semantic NLP")
    st.caption(
        "Pipeline: Sentence Transformer → Semantic Embeddings → "
        "Similarity Calculation → Skill Gap Analysis"
    )

    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader("Upload candidate resume (PDF)", type=["pdf"])
    with col2:
        jd_text_input = st.text_area("Paste the job description", height=220)

    if st.button("Analyze Match", type="primary"):
        if not resume_file or not jd_text_input.strip():
            st.warning("Please upload a resume PDF and paste a job description.")
            return

        with st.spinner("Loading model and analyzing (first run downloads the model)..."):
            model = load_model()
            resume_text = extract_text_from_pdf(resume_file)

            semantic_score = semantic_similarity(model, resume_text, jd_text_input)

            resume_skills = find_skills(resume_text)
            jd_skills = find_skills(jd_text_input)
            matched_skills = resume_skills & jd_skills
            missing_skills = jd_skills - resume_skills
            skill_coverage = (
                round(len(matched_skills) / len(jd_skills) * 100, 2) if jd_skills else 0.0
            )

        st.subheader("Result")
        c1, c2 = st.columns(2)
        c1.metric("Semantic Similarity", f"{semantic_score}%")
        c2.metric("Skill Coverage", f"{skill_coverage}%")

        st.markdown("**Matched skills:** " + (", ".join(sorted(matched_skills)) or "—"))
        if missing_skills:
            st.error("**Missing skills:** " + ", ".join(sorted(missing_skills)))
        else:
            st.success("No missing skills detected from the JD's required vocabulary.")

        with st.expander("View extracted resume text"):
            st.text(resume_text[:3000] or "No text could be extracted from this PDF.")


if __name__ == "__main__":
    main()
