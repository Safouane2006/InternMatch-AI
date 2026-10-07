import streamlit as st
import pandas as pd
import re

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="InternMatch AI",
    page_icon="💼",
    layout="wide"
)

data_path = Path(__file__).resolve().parent / "data"

df = pd.read_csv(data_path / "job_postings.csv")
skills = pd.read_csv(data_path / "job_skills.csv")
summary = pd.read_csv(data_path / "job_summary.csv")

df_merged = df.merge(skills, on="job_link", how="left")
df_merged = df_merged.merge(summary, on="job_link", how="left")

df_clean = df_merged[[
    "job_title",
    "company",
    "job_location",
    "job_level",
    "job_type",
    "job_skills",
    "job_summary",
    "job_link"
]].copy()

df_clean = df_clean.dropna(
    subset=["job_skills", "job_location"]
)

df_clean["matching_text"] = (
    df_clean["job_title"] + " " +
    df_clean["job_skills"] + " " +
    df_clean["job_summary"]
)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+#. ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df_clean["matching_text"] = (
    df_clean["matching_text"].apply(clean_text)
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

job_vectors = vectorizer.fit_transform(
    df_clean["matching_text"]
)


def recommend_jobs(profile, top_n=5):
    profile_clean = clean_text(profile)

    profile_vector = vectorizer.transform(
        [profile_clean]
    )

    scores = cosine_similarity(
        profile_vector,
        job_vectors
    ).flatten()

    results = df_clean.copy()
    results["match_score"] = (scores * 100).round(1)

    return results.sort_values(
        by="match_score",
        ascending=False
    )[[
        "job_title",
        "company",
        "job_location",
        "job_level",
        "job_type",
        "job_skills",
        "match_score",
        "job_link"
    ]].head(top_n)


st.title("InternMatch AI")

st.write(
    "Find job opportunities that match your skills and profile."
)

profile = st.text_area(
    "Describe your profile and skills:",
    height=180
)

top_n = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Find my matches"):
    if profile.strip():
        recommendations = recommend_jobs(
            profile,
            top_n
        )

        st.subheader("Best matches")

        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "Please describe your profile first."
        )
