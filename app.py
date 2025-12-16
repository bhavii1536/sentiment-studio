import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
from langdetect import detect

from transformers import pipeline
from googleapiclient.discovery import build

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Sentiment Analysis Studio",
    layout="wide"
)

PASTEL_COLORS = ["#AEC6CF", "#FFB7B2", "#B5EAD7", "#CDB4DB", "#E2F0CB"]

# ---------------- MODELS ----------------
@st.cache_resource
def load_models():
    sentiment_en = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    sentiment_multi = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )

    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

    return sentiment_en, sentiment_multi, emotion_model


sentiment_en, sentiment_multi, emotion_model = load_models()

# ---------------- YOUTUBE API ----------------
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# ---------------- HELPERS ----------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


def predict_sentiment(text):
    lang = detect_language(text)
    model = sentiment_en if lang == "en" else sentiment_multi
    label = model(text[:512])[0]["label"]

    if label in ["LABEL_0", "NEGATIVE"]:
        return "Negative"
    elif label in ["LABEL_1", "NEUTRAL"]:
        return "Neutral"
    else:
        return "Positive"


def detect_emotions(texts):
    emotions = []
    for t in texts:
        res = emotion_model(t[:512])[0]
        emotions.append(res["label"])
    return emotions


ASPECTS = {
    "Price": ["price", "cost", "expensive", "cheap"],
    "Quality": ["quality", "pure", "organic"],
    "Packaging": ["packaging", "bottle", "box"],
    "Smell": ["smell", "fragrance", "aroma"],
    "Delivery": ["delivery", "shipping"]
}


def aspect_based_sentiment(texts):
    rows = []
    for t in texts:
        tl = t.lower()
        for aspect, keys in ASPECTS.items():
            if any(k in tl for k in keys):
                rows.append({
                    "Aspect": aspect,
                    "Sentiment": predict_sentiment(t)
                })
    return pd.DataFrame(rows)


# ---------------- YOUTUBE FUNCTIONS ----------------
def fetch_comments(video_id, limit=100):
    comments = []
    req = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )
    res = req.execute()

    for item in res["items"][:limit]:
        comments.append(
            item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        )
    return comments


def search_videos(query, limit=3):
    res = youtube.search().list(
        q=query,
        part="id",
        type="video",
        maxResults=limit
    ).execute()

    return [i["id"]["videoId"] for i in res["items"]]


# ---------------- UI ----------------
st.title("📊 Sentiment Analysis Studio")
st.caption("Real-Time Media Opinion Analysis Using Machine Learning")

mode = st.selectbox(
    "Choose Analysis Type",
    [
        "Product / Topic Analysis (YouTube)",
        "YouTube Channel Insights",
        "CSV Upload Analysis"
    ]
)

# ---------- PRODUCT / TOPIC ----------
if mode == "Product / Topic Analysis (YouTube)":
    topic = st.text_input("Enter product / topic")

    if st.button("Analyze"):
        vids = search_videos(topic)
        all_comments = []

        for v in vids:
            all_comments.extend(fetch_comments(v))

        df = pd.DataFrame({
            "Text": all_comments,
            "Sentiment": [predict_sentiment(t) for t in all_comments]
        })

        st.subheader("📊 Sentiment Distribution")
        fig, ax = plt.subplots()
        df["Sentiment"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            colors=PASTEL_COLORS,
            ax=ax
        )
        ax.axis("equal")
        st.pyplot(fig)

        st.subheader("🧠 Aspect-Based Sentiment")
        absa = aspect_based_sentiment(all_comments)
        if not absa.empty:
            st.bar_chart(absa.value_counts().unstack().fillna(0))

        st.subheader("😊 Emotion Detection")
        emotions = detect_emotions(all_comments)
        emo_df = pd.Series(emotions).value_counts()

        fig2, ax2 = plt.subplots()
        emo_df.plot(kind="bar", color=PASTEL_COLORS, ax=ax2)
        st.pyplot(fig2)

# ---------- CHANNEL INSIGHTS ----------
elif mode == "YouTube Channel Insights":
    channel_name = st.text_input("Enter Channel Name")

    if st.button("Analyze Channel"):
        res = youtube.search().list(
            q=channel_name,
            part="snippet",
            type="channel",
            maxResults=1
        ).execute()

        channel_id = res["items"][0]["snippet"]["channelId"]

        stats = youtube.channels().list(
            part="statistics",
            id=channel_id
        ).execute()["items"][0]["statistics"]

        st.metric("Subscribers", stats["subscriberCount"])
        st.metric("Total Views", stats["viewCount"])

# ---------- CSV ----------
else:
    file = st.file_uploader("Upload CSV (text, label optional)", type="csv")

    if file:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("CSV must contain 'text' column")
        else:
            df["Sentiment"] = df["text"].apply(predict_sentiment)

            fig, ax = plt.subplots()
            df["Sentiment"].value_counts().plot(
                kind="bar",
                color=PASTEL_COLORS,
                ax=ax
            )
            st.pyplot(fig)
