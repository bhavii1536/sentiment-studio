import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
from transformers import pipeline
from googleapiclient.discovery import build

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Sentiment Analysis Studio",
    layout="wide"
)

# ==================================================
# GLOBAL CSS (PASTEL BG + EMOJI FRAME + ANIMATION)
# ==================================================
st.markdown("""
<style>
html, body {
    height: 100%;
}

/* Warm pastel background */
body {
    background: radial-gradient(circle at top, #fff4e6 0%, #f7e7d3 40%, #e8d5c4 100%);
}

/* App padding */
.block-container {
    padding-top: 2rem;
}

/* Emoji background layer */
.emoji-bg {
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
}

/* Floating animation */
@keyframes floatSlow {
    0%   { transform: translateY(0px) rotate(var(--r)); }
    50%  { transform: translateY(-14px) rotate(var(--r)); }
    100% { transform: translateY(0px) rotate(var(--r)); }
}

/* Emoji base */
.emoji {
    position: absolute;
    opacity: 0.28;
    animation: floatSlow 14s ease-in-out infinite;
}

/* Depth / size */
.small  { font-size: 48px;  filter: blur(1px);   }
.medium { font-size: 72px;  filter: blur(0.6px); }
.large  { font-size: 96px;  filter: blur(0.2px); }

/* Center content card */
.center-card {
    max-width: 920px;
    margin: auto;
    padding: 2.4rem;
    border-radius: 22px;
    background: rgba(20, 20, 20, 0.55);
    backdrop-filter: blur(10px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.25);
    color: #f5f5f5;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# EMOJI FRAME BACKGROUND
# ==================================================
st.markdown("""
<div class="emoji-bg">

    <!-- TOP -->
    <span class="emoji large"  style="top:3%; left:8%;  --r:-12deg;">😊</span>
    <span class="emoji medium" style="top:2%; left:28%; --r:8deg;">😍</span>
    <span class="emoji large"  style="top:4%; left:55%; --r:-6deg;">😄</span>
    <span class="emoji medium" style="top:3%; left:78%; --r:10deg;">😆</span>

    <!-- LEFT -->
    <span class="emoji medium" style="top:25%; left:2%; --r:-10deg;">🤔</span>
    <span class="emoji large"  style="top:48%; left:1%; --r:6deg;">😐</span>
    <span class="emoji medium" style="top:75%; left:3%; --r:-8deg;">😢</span>

    <!-- RIGHT -->
    <span class="emoji medium" style="top:22%; right:2%; --r:12deg;">😮</span>
    <span class="emoji large"  style="top:50%; right:1%; --r:-6deg;">😍</span>
    <span class="emoji medium" style="top:78%; right:3%; --r:10deg;">😡</span>

    <!-- BOTTOM -->
    <span class="emoji large"  style="bottom:3%; left:18%; --r:-8deg;">😊</span>
    <span class="emoji medium" style="bottom:2%; left:45%; --r:6deg;">😄</span>
    <span class="emoji large"  style="bottom:3%; left:72%; --r:-10deg;">😍</span>

</div>
""", unsafe_allow_html=True)

# ==================================================
# CENTER CONTENT START
# ==================================================
st.markdown('<div class="center-card">', unsafe_allow_html=True)

st.title("📊 Sentiment Analysis Studio")
st.caption("Real-Time Media Opinion Analysis Using Machine Learning")

COLORS = ["#4E79A7", "#F28E2B", "#E15759"]

# ==================================================
# LOAD MODELS
# ==================================================
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

# ==================================================
# YOUTUBE API
# ==================================================
youtube = build(
    "youtube",
    "v3",
    developerKey=st.secrets["YOUTUBE_API_KEY"]
)

# ==================================================
# HELPERS
# ==================================================
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

ASPECTS = {
    "Price": ["price", "cost", "expensive", "cheap"],
    "Quality": ["quality", "pure", "organic"],
    "Packaging": ["packaging", "bottle", "box"],
    "Delivery": ["delivery", "shipping"]
}

def aspect_based_sentiment(texts):
    rows = []
    for t in texts:
        tl = t.lower()
        for a, keys in ASPECTS.items():
            if any(k in tl for k in keys):
                rows.append({"Aspect": a, "Sentiment": predict_sentiment(t)})
    return pd.DataFrame(rows)

# ==================================================
# YOUTUBE DATA
# ==================================================
def search_videos(query, limit=3):
    res = youtube.search().list(
        q=query, part="id", type="video", maxResults=limit
    ).execute()
    return [i["id"]["videoId"] for i in res["items"]]

def fetch_comments(video_id, limit=80):
    res = youtube.commentThreads().list(
        part="snippet", videoId=video_id, maxResults=100
    ).execute()
    return [
        i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        for i in res["items"][:limit]
    ]

def fetch_channel_comments(channel_id):
    videos = youtube.search().list(
        channelId=channel_id, part="id", type="video", maxResults=3
    ).execute()
    comments = []
    for v in videos["items"]:
        comments.extend(fetch_comments(v["id"]["videoId"]))
    return comments

# ==================================================
# CHARTS
# ==================================================
def show_charts(sentiments):
    s = pd.Series(sentiments).value_counts()
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(3.8, 3.8), facecolor="none")
        ax.set_facecolor("none")
        ax.pie(
            s,
            labels=s.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=COLORS,
            wedgeprops={"width": 0.45}
        )
        ax.axis("equal")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(3.8, 3))
        ax.set_facecolor("none")
        s.plot(kind="barh", color=COLORS, ax=ax)
        ax.set_title("Sentiment Comparison")
        st.pyplot(fig)

# ==================================================
# MODE SELECTION
# ==================================================
mode = st.selectbox(
    "Choose Analysis Type",
    [
        "Product / Topic Analysis (YouTube)",
        "YouTube Channel Insights",
        "CSV Upload Analysis"
    ]
)

# ==================================================
# PRODUCT / TOPIC
# ==================================================
if mode == "Product / Topic Analysis (YouTube)":
    topic = st.text_input("Enter product / topic")

    if st.button("Analyze"):
        st.info(f"🔍 Analyzing public opinion on: {topic}")
        st.caption("Almost there… gathering real audience opinions")

        comments = []
        for v in search_videos(topic):
            comments.extend(fetch_comments(v))

        if comments:
            st.subheader("📄 Sample Comments")
            for i, c in enumerate(comments[:5], 1):
                st.write(f"{i}. {c}")

            sentiments = [predict_sentiment(c) for c in comments]
            show_charts(sentiments)

            st.subheader("🧠 Aspect-Based Sentiment")
            absa = aspect_based_sentiment(comments)
            if not absa.empty:
                st.bar_chart(absa.value_counts().unstack().fillna(0))

# ==================================================
# CHANNEL INSIGHTS
# ==================================================
elif mode == "YouTube Channel Insights":
    channel = st.text_input("Enter Channel Name")

    if st.button("Analyze Channel"):
        res = youtube.search().list(
            q=channel, part="snippet", type="channel", maxResults=1
        ).execute()

        if res["items"]:
            cid = res["items"][0]["snippet"]["channelId"]
            stats = youtube.channels().list(
                part="statistics", id=cid
            ).execute()["items"][0]["statistics"]

            m1, m2 = st.columns(2)
            m1.metric("Subscribers", stats["subscriberCount"])
            m2.metric("Total Views", stats["viewCount"])

            comments = fetch_channel_comments(cid)
            if comments:
                sentiments = [predict_sentiment(c) for c in comments]
                show_charts(sentiments)

                st.subheader("📄 Sample Audience Comments")
                for i, c in enumerate(comments[:5], 1):
                    st.write(f"{i}. {c}")

# ==================================================
# CSV ANALYSIS
# ==================================================
else:
    file = st.file_uploader("Upload CSV (text column required)", type="csv")
    if file:
        df = pd.read_csv(file)
        if "text" in df.columns:
            sentiments = df["text"].apply(predict_sentiment)
            show_charts(sentiments)
        else:
            st.error("CSV must contain a 'text' column")

# ==================================================
# END CENTER CARD
# ==================================================
st.markdown('</div>', unsafe_allow_html=True)
