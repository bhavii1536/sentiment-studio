import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from langdetect import detect
from googleapiclient.discovery import build

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Sentiment Analysis Studio", layout="wide")

# ===============================
# CSS
# ===============================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 14px;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# CENTER LAYOUT
# ===============================
_, main_col, _ = st.columns([1, 4, 1])

with main_col:
    st.title("📊 Sentiment Analysis Studio")
    st.caption("Real-Time Media Opinion Analysis Using Machine Learning")

    COLORS = ["#2563EB", "#F97316"]  # blue, orange

    # ===============================
    # LOAD MODEL (FAST)
    # ===============================
    @st.cache_resource
    def load_model():
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    sentiment_model = load_model()

    # ===============================
    # SENTIMENT FUNCTION
    # ===============================
    def predict_sentiment(text):
        try:
            res = sentiment_model(text[:512])[0]["label"]
            return "Positive" if res == "POSITIVE" else "Negative"
        except:
            return "Negative"

    # ===============================
    # CHARTS
    # ===============================
    def show_charts(sentiments):
        counts = pd.Series(sentiments).value_counts()

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ax.pie(
                counts,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=COLORS
            )
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(4, 3))
            counts.plot(kind="bar", color=COLORS, ax=ax)
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Comparison")
            st.pyplot(fig)

    # ===============================
    # MODE SELECT
    # ===============================
    mode = st.selectbox(
        "Choose Analysis Type",
        ["CSV Upload Analysis"]
    )

    # ===============================
    # CSV UPLOAD ANALYSIS (FIXED)
    # ===============================
    if mode == "CSV Upload Analysis":
        file = st.file_uploader("Upload CSV", type=["csv"])

        if file:
            analyze = st.button("Analyze Dataset")

            if analyze:
                # 🔹 SAFE CSV READ
                try:
                    df = pd.read_csv(file, encoding="utf-8")
                except:
                    df = pd.read_csv(file, encoding="latin1")

                st.success(f"CSV loaded: {len(df)} rows")

                st.write("Detected columns:")
                st.code(list(df.columns))

                # 🔹 AUTO-DETECT TEXT COLUMN
                TEXT_CANDIDATES = [
                    "text", "tweet", "review", "comment", "content", "message"
                ]

                text_col = None

                for col in df.columns:
                    if col.lower() in TEXT_CANDIDATES:
                        text_col = col
                        break

                # fallback: first string column
                if not text_col:
                    for col in df.columns:
                        if df[col].dtype == "object":
                            text_col = col
                            break

                if not text_col:
                    st.error("❌ No text-like column found.")
                    st.stop()

                st.success(f"Using text column: `{text_col}`")

                texts = df[text_col].astype(str).dropna().tolist()

                # 🔹 LIMIT for SPEED
                MAX_SAMPLES = min(2000, len(texts))
                texts = texts[:MAX_SAMPLES]

                st.info(f"Analyzing {len(texts)} rows…")

                sentiments = [predict_sentiment(t) for t in texts]

                show_charts(sentiments)

                st.subheader("📄 Sample Texts")
                for i, t in enumerate(texts[:5], 1):
                    st.write(f"{i}. {t}")
