import streamlit as st
import pandas as pd
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

# ===============================
# GLOBAL CSS (HERO + CARDS + EMOJI ATMOSPHERE)
# ===============================
st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Remove default padding look */
.block-container {
    padding-top: 0.5rem;
}

/* HERO */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem 1rem;
    position: relative;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff9a9e, #fad0c4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    font-size: 1.15rem;
    color: #d1d5db;
    margin-top: 0.6rem;
}

/* Floating cards */
.card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem auto;
    max-width: 850px;
    box-shadow: 0 25px 45px rgba(0,0,0,0.45);
}

/* Section titles */
.section-title {
    font-size: 1.4rem;
    margin-bottom: 1rem;
}

/* EMOJI ATMOSPHERE */
.emoji-bg {
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
}

@keyframes float {
    0% { transform: translateY(0px) rotate(var(--r)); }
    50% { transform: translateY(-14px) rotate(var(--r)); }
    100% { transform: translateY(0px) rotate(var(--r)); }
}

.emoji {
    position: absolute;
    opacity: 0.18;
    animation: float 14s ease-in-out infinite;
}

.small  { font-size: 48px; filter: blur(1px); }
.medium { font-size: 72px; filter: blur(0.6px); }
.large  { font-size: 96px; filter: blur(0.2px); }

</style>
""", unsafe_allow_html=True)

# ===============================
# EMOJI BACKGROUND (ATMOSPHERE ONLY)
# ===============================
st.markdown("""
<div class="emoji-bg">
    <span class="emoji large"  style="top:5%; left:8%;  --r:-12deg;">😊</span>
    <span class="emoji medium" style="top:12%; left:70%; --r:10deg;">😍</span>
    <span class="emoji small"  style="top:30%; left:15%; --r:-8deg;">😐</span>
    <span class="emoji large"  style="top:45%; left:85%; --r:12deg;">😮</span>
    <span class="emoji medium" style="top:65%; left:6%;  --r:-10deg;">😡</span>
    <span class="emoji small"  style="top:80%; left:75%; --r:8deg;">🤔</span>
    <span class="emoji large"  style="top:88%; left:40%; --r:-6deg;">😄</span>
</div>
""", unsafe_allow_html=True)

# ===============================
# HERO SECTION
# ===============================
st.markdown("""
<div class="hero">
    <div class="hero-title">Sentiment Analysis Studio</div>
    <div class="hero-sub">
        Decode public opinion from YouTube using Machine Learning
    </div>
</div>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    en = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    multi = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )
    return en, multi

sentiment_en, sentiment_multi = load_models()

# ===============================
# YOUTUBE API
# ===============================
youtube = build(
    "youtube",
    "v3",
    developerKey=st.secrets["YOUTUBE_API_KEY"]
)

# ===============================
# HELPERS
# ===============================
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def predict_sentiment(text):
    model = sentiment_en if detect_language(text) == "en" else sentiment_multi
    label = model(text[:512])[0]["label"]
    if label in ["LABEL_0", "NEGATIVE"]:
        return "Negative"
    elif label in ["LABEL_1", "NEUTRAL"]:
        return "Neutral"
    return "Positive"

# ===============================
# YOUTUBE DATA
# ===============================
def search_videos(query, limit=3):
    res = youtube.search().list(
        q=query, part="id", type="video", maxResults=limit
    ).execute()
    return [i["id"]["videoId"] for i in res["items"]]

def fetch_comments(video_id, limit=60):
    res = youtube.commentThreads().list(
        part="snippet", videoId=video_id, maxResults=100
    ).execute()
    return [
        i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        for i in res["items"][:limit]
    ]

# ===============================
# INPUT CARD
# ===============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔍 Start Analysis</div>', unsafe_allow_html=True)

topic = st.text_input("Enter product / brand / topic")
analyze = st.button("✨ Analyze Sentiment")

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# RESULTS
# ===============================
if analyze and topic:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"🔍 **Analyzing public opinion on:** `{topic}`")

    comments = []
    for v in search_videos(topic):
        comments.extend(fetch_comments(v))

    if comments:
        sentiments = [predict_sentiment(c) for c in comments]

        st.markdown('<div class="section-title">📊 Sentiment Overview</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4,4), facecolor="none")
        ax.set_facecolor("none")
        pd.Series(sentiments).value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            startangle=90,
            ax=ax
        )
        ax.axis("equal")
        st.pyplot(fig)

        st.markdown('<div class="section-title">📄 Sample Comments</div>', unsafe_allow_html=True)
        for i, c in enumerate(comments[:5], 1):
            st.write(f"{i}. {c}")

    st.markdown('</div>', unsafe_allow_html=True)
