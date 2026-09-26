"""
Level 3 — Production-style Application
=======================================
Pipeline:
    Resume PDF -> Text Extraction -> NLP Preprocessing
        -> [Skill NLP, Embeddings] -> Matching Engine
        -> [Match Score, Skill Gaps] -> Streamlit UI

Run:
    streamlit run level3_production/app.py
"""

import sys
import os

import streamlit as st

sys.path.append(os.path.dirname(__file__))

from utils.pdf_parser import extract_text_from_pdf
from utils.matching_engine import MatchingEngine
from utils.visualization import score_bar_chart, skills_donut_chart


@st.cache_resource(show_spinner=False)
def get_engine() -> MatchingEngine:
    """Load the matching engine (and its embedding model) once per session."""
    return MatchingEngine()


def recommendation_badge(recommendation: str) -> str:
    colors = {"Shortlist": "🟢", "Consider": "🟡", "Reject": "🔴"}
    return f"{colors.get(recommendation, '⚪')} **{recommendation}**"


def main():
    st.set_page_config(
        page_title="AI Resume Screener — Level 3", page_icon="🎯", layout="wide"
    )

    st.title("🎯 AI-Powered Resume Screening & Job Matching")
    st.caption(
        "Resume PDF → Text Extraction → NLP Preprocessing → Skill NLP + "
        "Embeddings → Matching Engine → Match Score & Skill Gaps"
    )

    with st.sidebar:
        st.header("About")
        st.write(
            "This is the production-style Level 3 build. It combines a "
            "skills taxonomy, sentence-transformer embeddings, experience "
            "heuristics, and education matching into one weighted score."
        )
        st.markdown("**Scoring weights**")
        st.write("- Technical Skills: 40%")
        st.write("- Semantic Similarity: 30%")
        st.write("- Experience Relevance: 20%")
        st.write("- Education Match: 10%")

    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader("📄 Upload candidate resume (PDF)", type=["pdf"])
    with col2:
        jd_text = st.text_area("💼 Paste the job description", height=260)

    analyze_clicked = st.button("Analyze Candidate", type="primary", use_container_width=True)

    if not analyze_clicked:
        return

    if not resume_file or not jd_text.strip():
        st.warning("Please upload a resume PDF and paste a job description.")
        return

    with st.spinner("Extracting resume text..."):
        resume_text = extract_text_from_pdf(resume_file)

    if not resume_text.strip():
        st.error("Couldn't extract any text from this PDF. Try a different file.")
        return

    with st.spinner("Loading models and running the matching engine..."):
        engine = get_engine()
        result = engine.analyze(resume_text, jd_text)

    # ---------------------------------------------------------------- #
    # Headline result — mirrors the example output format
    # ---------------------------------------------------------------- #
    st.markdown("---")
    top = st.columns([1, 1, 1, 1, 1])
    top[0].metric("Match Score", f"{result.match_score:.0f}%")
    top[1].metric("Technical Skills", f"{result.technical_skills_score:.0f}%")
    top[2].metric("Experience Relevance", f"{result.experience_score:.0f}%")
    top[3].metric("Education", f"{result.education_score:.0f}%")
    with top[4]:
        st.write("**Recommendation**")
        st.markdown(recommendation_badge(result.recommendation))

    missing = ", ".join(sorted(result.missing_skills)) or "None 🎉"
    st.markdown(f"**Missing Skills:** {missing}")

    # ---------------------------------------------------------------- #
    # Charts
    # ---------------------------------------------------------------- #
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.pyplot(score_bar_chart(result), use_container_width=True)
    with chart_col2:
        donut = skills_donut_chart(result)
        if donut is not None:
            st.pyplot(donut, use_container_width=True)
        else:
            st.info("No skills detected in either document to chart.")

    # ---------------------------------------------------------------- #
    # Details
    # ---------------------------------------------------------------- #
    st.markdown("---")
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.subheader("Skills")
        st.write("**Matched:** " + (", ".join(sorted(result.matched_skills)) or "—"))
        st.write("**Missing:** " + (", ".join(sorted(result.missing_skills)) or "—"))

    with detail_col2:
        st.subheader("Experience & Education")
        st.write(f"**Candidate experience (detected):** {result.candidate_years:.1f} years")
        st.write(f"**Required experience (detected):** {result.required_years:.1f} years")
        st.write(f"**Candidate education (detected):** {result.candidate_education}")
        st.write(f"**Required education (detected):** {result.required_education}")

    with st.expander("View extracted resume text"):
        st.text(resume_text[:5000])


if __name__ == "__main__":
    main()
