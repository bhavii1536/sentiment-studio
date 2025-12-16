# ==================================================
# Sentiment Studio
# Multilingual Media Insight Dashboard
# ==================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from langdetect import detect
from transformers import pipeline

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Sentiment Studio",
    layout="centered"
)

# -------------------------
# TITLE
# -------------------------
st.title("📊 Sentiment Studio")
st.subheader("Multilingual Media Insight Dashboard")

st.write(
    "Analyze sentiment from datasets and social media content "
    "in **English, Tamil, and Hindi**."
)

st.markdown("---")

# -------------------------
# LOAD MODELS (CACHED)
# -------------------------
@st.cache_resource
def load_models():
    sentiment_en = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )
    sentiment_multi = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )
    return sentiment_en, sentiment_multi


sentiment_en, sentiment_multi = load_models()

# -------------------------
# LABEL MAP
# -------------------------
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive"
}

# -------------------------
# SENTIMENT FUNCTION
# -------------------------
def predict_sentiment(text):
    if not text or str(text).strip() == "":
        return "Neutral"

    text = str(text)

    try:
        lang = detect(text)
    except:
        lang = "en"

    try:
        if lang == "en":
            out = sentiment_en(text[:512])[0]
        else:
            out = sentiment_multi(text[:512])[0]

        return label_map[out["label"]]
    except:
        return "Neutral"

# -------------------------
# INPUT TYPE
# -------------------------
input_type = st.selectbox(
    "Choose input type:",
    ["CSV File Upload"]
)

st.markdown("### Upload CSV")

uploaded_file = st.file_uploader(
    "Upload CSV file (must contain a column named `text`)",
    type=["csv"]
)

st.markdown("---")

# -------------------------
# ANALYSIS
# -------------------------
if st.button("Analyze"):

    if uploaded_file is None:
        st.warning("⚠️ Please upload a CSV file.")
    else:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("❌ CSV must contain a column named `text`")
        else:
            st.info("⏳ Performing sentiment analysis...")

            df["Sentiment"] = df["text"].astype(str).apply(predict_sentiment)

            st.success("✅ Analysis completed")

            # Show data
            st.subheader("📄 Sample Results")
            st.dataframe(df.head(20))

            # Pie chart
            st.subheader("📊 Overall Sentiment Distribution")
            counts = df["Sentiment"].value_counts()

            fig, ax = plt.subplots()
            ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=90
            )
            ax.axis("equal")
            st.pyplot(fig)

            # Download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Results",
                csv,
                "sentiment_results.csv",
                "text/csv"
            )

st.markdown("---")
st.caption("Sentiment Studio | Phase 1 Completed")
