"""
Chart helpers for the Level 3 Streamlit app.
"""

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def score_bar_chart(result):
    """Horizontal bar chart of the four score components."""
    labels = ["Technical Skills", "Semantic Match", "Experience", "Education"]
    values = [
        result.technical_skills_score,
        result.semantic_score,
        result.experience_score,
        result.education_score,
    ]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    colors = sns.color_palette("viridis", len(labels))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (%)")
    ax.set_title("Match Score Breakdown")

    for bar, value in zip(bars, values):
        ax.text(
            min(value + 2, 96), bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}%", va="center", fontsize=9,
        )

    fig.tight_layout()
    return fig


def skills_donut_chart(result):
    """Donut chart of matched vs missing skill counts."""
    matched_n = len(result.matched_skills)
    missing_n = len(result.missing_skills)

    if matched_n == 0 and missing_n == 0:
        return None

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, _texts, autotexts = ax.pie(
        [matched_n, missing_n],
        labels=["Matched", "Missing"],
        autopct=lambda p: f"{int(round(p / 100 * (matched_n + missing_n)))}",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
        wedgeprops=dict(width=0.4),
    )
    ax.set_title("Skill Coverage")
    fig.tight_layout()
    return fig
