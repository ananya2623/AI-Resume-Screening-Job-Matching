"""
Level 1 — Basic ML
==================
Pipeline:
    Resume PDF -> Text Extraction -> Cleaning & Preprocessing
                -> TF-IDF -> Cosine Similarity -> Match Score

Run:
    streamlit run level1_basic_ml/app_level1.py
"""

import re
import string

import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# 1. Text extraction
# --------------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF file using pdfplumber."""
    text_chunks = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


# --------------------------------------------------------------------------
# 2. Cleaning & preprocessing
# --------------------------------------------------------------------------
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "as", "by", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "from", "will", "we", "you",
    "your", "our", "their", "i", "he", "she", "they", "them", "us", "have",
    "has", "had", "do", "does", "did", "not", "so", "than", "then", "there",
}


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/numbers/extra whitespace, remove stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


# --------------------------------------------------------------------------
# 3. TF-IDF + Cosine similarity
# --------------------------------------------------------------------------
def compute_match_score(resume_text: str, jd_text: str) -> float:
    """Vectorize both documents with TF-IDF and return cosine similarity (%)."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Resume Matcher — Level 1 (Basic ML)", page_icon="📄")
    st.title("📄 Resume ↔ Job Match — Level 1: Basic ML")
    st.caption("Pipeline: Text Extraction → Cleaning → TF-IDF → Cosine Similarity")

    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader("Upload candidate resume (PDF)", type=["pdf"])
    with col2:
        jd_text_input = st.text_area("Paste the job description", height=220)

    if st.button("Analyze Match", type="primary"):
        if not resume_file or not jd_text_input.strip():
            st.warning("Please upload a resume PDF and paste a job description.")
            return

        with st.spinner("Extracting and analyzing..."):
            raw_resume_text = extract_text_from_pdf(resume_file)
            cleaned_resume = clean_text(raw_resume_text)
            cleaned_jd = clean_text(jd_text_input)
            match_score = compute_match_score(cleaned_resume, cleaned_jd)

        st.subheader("Result")
        st.metric("Match Score", f"{match_score}%")
        st.progress(min(int(match_score), 100))

        with st.expander("View extracted resume text"):
            st.text(raw_resume_text[:3000] or "No text could be extracted from this PDF.")


if __name__ == "__main__":
    main()
