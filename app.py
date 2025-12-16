import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langdetect import detect
from transformers import pipeline
from googleapiclient.discovery import build

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Sentiment Analysis Studio",
    layout="wide"
)

st.title("📊 Sentiment Analysis Studio")
st.caption("Real-Time Media Opinion Analysis Using Machine Learning")

PASTEL_COLORS = ["#AEC6CF", "#FFB7B2", "#B5EAD7", "#CDB4DB", "#E2F0CB"]

# ===============================
# LOAD MODELS
# ===============================
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

# ===============================
# YOUTUBE API
# ===============================
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# ===============================
# HELPER FUNCTIONS
# ===============================
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
        results = emotion_model(t[:512])
        top_emotion = max(results, key=lambda x: x["score"])
        emotions.append(top_emotion["label"].capitalize())
    return emotions


ASPECTS = {
    "Price": ["price", "cost", "expensive", "cheap"],
    "Quality": ["quality", "pure", "organic", "good", "bad"],
    "Packaging": ["packaging", "bottle", "box"],
    "Smell": ["smell", "fragrance", "aroma"],
    "Delivery": ["delivery", "shipping", "late"]
}


def aspect_based_sentiment(texts):
    rows = []
    for t in texts:
        t_low = t.lower()
        for aspect, keys in ASPECTS.items():
            if any(k in t_low for k in keys):
                rows.append({
                    "Aspect": aspect,
                    "Sentiment": predict_sentiment(t)
                })
    return pd.DataFrame(rows)


# ===============================
# YOUTUBE FUNCTIONS
# ===============================
def search_videos(query, limit=3):
    res = youtube.search().list(
        q=query,
        part="id",
        type="video",
        maxResults=limit
    ).execute()

    return [i["id"]["videoId"] for i in res["items"]]


def fetch_comments(video_id, limit=100):
    comments = []
    req = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    ).execute()

    for item in req["items"][:limit]:
        comments.append(
            item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        )
    return comments


def fetch_channel_recent_comments(channel_id, limit=3):
    videos = youtube.search().list(
        channelId=channel_id,
        part="id",
        type="video",
        maxResults=limit
    ).execute()

    comments = []
    for v in videos["items"]:
        comments.extend(fetch_comments(v["id"]["videoId"], limit=50))
    return comments


# ===============================
# UI MODE
# ===============================
mode = st.selectbox(
    "Choose Analysis Type",
    [
        "Product / Topic Analysis (YouTube)",
        "YouTube Channel Insights",
        "CSV Upload Analysis"
    ]
)

# ===============================
# PRODUCT / TOPIC ANALYSIS
# ===============================
if mode == "Product / Topic Analysis (YouTube)":
    topic = st.text_input("Enter product / topic")

    if st.button("Analyze"):
        st.info(f"🔍 Analyzing public opinion on: **{topic}**")
        st.caption("✨ Almost there… gathering real voices from YouTube")

        with st.spinner("📥 Collecting comments and running ML models..."):
            videos = search_videos(topic)
            all_comments = []
            for v in videos:
                all_comments.extend(fetch_comments(v))

        if not all_comments:
            st.error("No comments found.")
        else:
            # ---- SAMPLE COMMENTS ----
            st.subheader("📄 Sample Comments")
            for i, c in enumerate(all_comments[:5], 1):
                st.write(f"**{i}.** {c}")

            sentiments = [predict_sentiment(t) for t in all_comments]
            df = pd.DataFrame({"Sentiment": sentiments})

            # ---- PIE CHART ----
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

            # ---- ASPECT BASED ----
            st.subheader("🧠 Aspect-Based Sentiment")
            absa = aspect_based_sentiment(all_comments)
            if not absa.empty:
                st.bar_chart(absa.value_counts().unstack().fillna(0))

            # ---- EMOTION ----
            st.subheader("😊 Emotion Detection")
            emotions = detect_emotions(all_comments)
            emo_df = pd.Series(emotions).value_counts()

            fig2, ax2 = plt.subplots()
            emo_df.plot(kind="bar", color=PASTEL_COLORS, ax=ax2)
            st.pyplot(fig2)

# ===============================
# CHANNEL INSIGHTS
# ===============================
elif mode == "YouTube Channel Insights":
    channel_name = st.text_input("Enter Channel Name")

    if st.button("Analyze Channel"):
        with st.spinner("Fetching channel data..."):
            res = youtube.search().list(
                q=channel_name,
                part="snippet",
                type="channel",
                maxResults=1
            ).execute()

        if not res["items"]:
            st.error("Channel not found.")
        else:
            channel_id = res["items"][0]["snippet"]["channelId"]

            stats = youtube.channels().list(
                part="statistics",
                id=channel_id
            ).execute()["items"][0]["statistics"]

            col1, col2 = st.columns(2)
            col1.metric("Subscribers", stats["subscriberCount"])
            col2.metric("Total Views", stats["viewCount"])

            st.subheader("📊 Audience Sentiment (Recent Videos)")
            with st.spinner("Analyzing audience reactions..."):
                channel_comments = fetch_channel_recent_comments(channel_id)

            if channel_comments:
                sentiments = [predict_sentiment(c) for c in channel_comments]
                sent_df = pd.Series(sentiments).value_counts()

                # PIE
                fig1, ax1 = plt.subplots()
                sent_df.plot(
                    kind="pie",
                    autopct="%1.1f%%",
                    colors=PASTEL_COLORS,
                    ax=ax1
                )
                ax1.set_ylabel("")
                st.pyplot(fig1)

                # BAR
                fig2, ax2 = plt.subplots()
                sent_df.plot(
                    kind="bar",
                    color=PASTEL_COLORS,
                    ax=ax2
                )
                st.pyplot(fig2)

                st.subheader("📄 Sample Audience Comments")
                for i, c in enumerate(channel_comments[:5], 1):
                    st.write(f"**{i}.** {c}")
            else:
                st.warning("No recent comments found.")

# ===============================
# CSV UPLOAD
# ===============================
else:
    file = st.file_uploader("Upload CSV (must contain 'text' column)", type="csv")

    if file:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            df["Sentiment"] = df["text"].apply(predict_sentiment)

            st.subheader("📊 Sentiment Distribution")
            fig, ax = plt.subplots()
            df["Sentiment"].value_counts().plot(
                kind="bar",
                color=PASTEL_COLORS,
                ax=ax
            )
            st.pyplot(fig)
