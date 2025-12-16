# ==================================================
# Sentiment Analysis Studio
# Real-Time Media Opinion Analysis Using ML
# ==================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from langdetect import detect
from transformers import pipeline
from googleapiclient.discovery import build

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Sentiment Analysis Studio",
    layout="centered"
)

# -------------------------
# TITLE
# -------------------------
st.title("📊 Sentiment Analysis Studio")
st.subheader("Real-Time Media Opinion Analysis Using Machine Learning")

st.write(
    "Analyze public opinion from **datasets and YouTube** "
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
# YOUTUBE API FUNCTIONS
# -------------------------
def get_youtube_client():
    return build(
        "youtube",
        "v3",
        developerKey=st.secrets["YOUTUBE_API_KEY"]
    )

def search_youtube_videos(query, max_results=3):
    youtube = get_youtube_client()
    request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=max_results
    )
    response = request.execute()

    return [
        item["id"]["videoId"]
        for item in response.get("items", [])
    ]

def fetch_youtube_comments(video_id, max_comments=50):
    youtube = get_youtube_client()
    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )
    response = request.execute()

    for item in response.get("items", []):
        text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(text)

        if len(comments) >= max_comments:
            break

    return comments

def analyze_product_youtube(topic):
    video_ids = search_youtube_videos(topic)
    all_comments = []

    for vid in video_ids:
        all_comments.extend(fetch_youtube_comments(vid))

    results = []
    for text in all_comments:
        sentiment = predict_sentiment(text)
        results.append({
            "Comment": text,
            "Sentiment": sentiment
        })

    return pd.DataFrame(results)

# -------------------------
# INPUT TYPE
# -------------------------
input_type = st.selectbox(
    "Choose analysis type:",
    [
        "CSV File Upload",
        "Product / Topic Analysis (YouTube)"
    ]
)

st.markdown("---")

# -------------------------
# CSV INPUT
# -------------------------
uploaded_file = None
topic_input = None

if input_type == "CSV File Upload":
    uploaded_file = st.file_uploader(
        "Upload CSV file (must contain `text` column)",
        type=["csv"]
    )

elif input_type == "Product / Topic Analysis (YouTube)":
    topic_input = st.text_input(
        "Enter product name / hashtag / topic",
        placeholder="e.g., iPhone 16, #BiggBossTamil"
    )

st.markdown("---")

# -------------------------
# ANALYZE BUTTON
# -------------------------
if st.button("Analyze"):

    # ---------- CSV ANALYSIS ----------
    if input_type == "CSV File Upload":

        if uploaded_file is None:
            st.warning("⚠️ Please upload a CSV file.")
        else:
            df = pd.read_csv(uploaded_file)

            if "text" not in df.columns:
                st.error("❌ CSV must contain a column named `text`.")
            else:
                st.info("⏳ Performing sentiment analysis...")

                df["Sentiment"] = df["text"].astype(str).apply(predict_sentiment)

                st.success("✅ Analysis completed")

                st.subheader("📄 Sample Results")
                st.dataframe(df.head(20))

                st.subheader("📊 Sentiment Distribution")
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

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download Results",
                    csv,
                    "sentiment_results.csv",
                    "text/csv"
                )

    # ---------- PRODUCT / TOPIC ANALYSIS ----------
    elif input_type == "Product / Topic Analysis (YouTube)":

        if not topic_input:
            st.warning("⚠️ Please enter a product or topic.")
        else:
            st.info(f"🔍 Fetching YouTube opinions on: {topic_input}")

            df = analyze_product_youtube(topic_input)

            if df.empty:
                st.error("❌ No comments found.")
            else:
                st.success("✅ YouTube analysis completed")

                st.subheader("📄 Sample Comments")
                st.dataframe(df.head(20))

                st.subheader("📊 Sentiment Distribution (YouTube)")
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

                pos = counts.get("Positive", 0)
                total = len(df)
                pct = round((pos / total) * 100, 1)

                st.markdown("### 🧠 Insight Summary")
                st.write(
                    f"Public opinion about **{topic_input}** on YouTube is "
                    f"**{pct}% positive**, indicating overall audience sentiment."
                )

st.markdown("---")
st.caption("Sentiment Analysis Studio | Real-Time Media Opinion Analysis Using ML")
