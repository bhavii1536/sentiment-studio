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
import snscrape.modules.twitter as sntwitter

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(page_title="Sentiment Analysis Studio", layout="centered")

st.title("📊 Sentiment Analysis Studio")
st.subheader("Real-Time Media Opinion Analysis Using Machine Learning")
st.write(
    "Analyze public opinion from **CSV datasets, YouTube, and Twitter** "
    "in **English, Tamil, and Hindi**."
)
st.markdown("---")

# ==================================================
# PASTEL COLORS
# ==================================================
PASTEL_COLORS = {
    "Positive": "#A8E6CF",
    "Neutral": "#FFD3B6",
    "Negative": "#FFAAA5"
}

# ==================================================
# LOAD MODELS (CACHED)
# ==================================================
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

# ==================================================
# SENTIMENT FUNCTION
# ==================================================
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
            out = sentiment_en(text[:512])[0]["label"]
        else:
            out = sentiment_multi(text[:512])[0]["label"]
    except:
        return "Neutral"

    return {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
        "positive": "Positive"
    }.get(out, "Neutral")

# ==================================================
# CHART HELPERS
# ==================================================
def pastel_pie_chart(counts, title):
    colors = [PASTEL_COLORS.get(k, "#D3D3D3") for k in counts.index]
    fig, ax = plt.subplots()
    ax.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    ax.set_title(title)
    st.pyplot(fig)

def pastel_bar_chart(df, title):
    fig, ax = plt.subplots()
    df.plot(kind="bar", ax=ax, color=["#B5EAD7", "#C7CEEA"])
    ax.set_title(title)
    ax.set_ylabel("Count")
    st.pyplot(fig)

# ==================================================
# YOUTUBE API
# ==================================================
def get_youtube_client():
    return build("youtube", "v3", developerKey=st.secrets["YOUTUBE_API_KEY"])

def search_youtube_videos(query, max_results=3):
    yt = get_youtube_client()
    req = yt.search().list(q=query, part="snippet", type="video", maxResults=max_results)
    res = req.execute()
    return [item["id"]["videoId"] for item in res.get("items", [])]

def fetch_youtube_comments(video_id, max_comments=50):
    yt = get_youtube_client()
    comments = []
    req = yt.commentThreads().list(
        part="snippet", videoId=video_id, maxResults=100, textFormat="plainText"
    )
    res = req.execute()

    for item in res.get("items", []):
        comments.append(
            item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        )
        if len(comments) >= max_comments:
            break
    return comments

def analyze_product_youtube(topic):
    vids = search_youtube_videos(topic)
    data = []
    for v in vids:
        for c in fetch_youtube_comments(v):
            data.append({"Text": c, "Sentiment": predict_sentiment(c)})
    return pd.DataFrame(data)

# ==================================================
# TWITTER (snscrape)
# ==================================================
def analyze_product_twitter(topic, limit=100):
    data = []
    for i, tweet in enumerate(
        sntwitter.TwitterSearchScraper(topic).get_items()
    ):
        if i >= limit:
            break
        data.append({
            "Text": tweet.content,
            "Sentiment": predict_sentiment(tweet.content)
        })
    return pd.DataFrame(data)

# ==================================================
# YOUTUBE CHANNEL INSIGHTS
# ==================================================
def get_channel_id(channel_name):
    yt = get_youtube_client()
    req = yt.search().list(
        q=channel_name, part="snippet", type="channel", maxResults=1
    )
    res = req.execute()
    if not res["items"]:
        return None
    return res["items"][0]["id"]["channelId"]

def get_channel_stats(channel_id):
    yt = get_youtube_client()
    req = yt.channels().list(part="statistics", id=channel_id)
    res = req.execute()
    stats = res["items"][0]["statistics"]
    return {
        "Subscribers": int(stats.get("subscriberCount", 0)),
        "Total Views": int(stats.get("viewCount", 0)),
        "Total Videos": int(stats.get("videoCount", 0))
    }

def get_recent_videos(channel_id, max_results=3):
    yt = get_youtube_client()
    req = yt.search().list(
        part="id", channelId=channel_id,
        order="date", type="video", maxResults=max_results
    )
    res = req.execute()
    return [i["id"]["videoId"] for i in res["items"]]

def analyze_channel_sentiment(channel_id):
    data = []
    for vid in get_recent_videos(channel_id):
        for c in fetch_youtube_comments(vid):
            data.append({"Text": c, "Sentiment": predict_sentiment(c)})
    return pd.DataFrame(data)

# ==================================================
# INPUT TYPE
# ==================================================
input_type = st.selectbox(
    "Choose analysis type:",
    [
        "CSV File Upload",
        "Product / Topic Analysis",
        "YouTube Channel Insights"
    ]
)

st.markdown("---")

# ==================================================
# INPUT UI
# ==================================================
uploaded_file = None
topic_input = None
channel_name = None

if input_type == "CSV File Upload":
    uploaded_file = st.file_uploader("Upload CSV (must contain 'text' column)", type=["csv"])

elif input_type == "Product / Topic Analysis":
    topic_input = st.text_input("Enter product / topic / hashtag")

elif input_type == "YouTube Channel Insights":
    channel_name = st.text_input("Enter YouTube channel name")

# ==================================================
# ANALYZE BUTTON
# ==================================================
if st.button("Analyze"):

    # ---------- CSV ----------
    if input_type == "CSV File Upload":
        if not uploaded_file:
            st.warning("Upload a CSV file")
        else:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("CSV must contain 'text' column")
            else:
                df["Sentiment"] = df["text"].astype(str).apply(predict_sentiment)
                st.dataframe(df.head(20))
                pastel_pie_chart(df["Sentiment"].value_counts(), "Sentiment Distribution")

    # ---------- PRODUCT / TOPIC ----------
    elif input_type == "Product / Topic Analysis":
        yt_df = analyze_product_youtube(topic_input)
        tw_df = analyze_product_twitter(topic_input)

        st.subheader("📊 YouTube Sentiment")
        pastel_pie_chart(yt_df["Sentiment"].value_counts(), "YouTube")

        st.subheader("📊 Twitter Sentiment")
        pastel_pie_chart(tw_df["Sentiment"].value_counts(), "Twitter")

        comp = pd.DataFrame({
            "YouTube": yt_df["Sentiment"].value_counts(),
            "Twitter": tw_df["Sentiment"].value_counts()
        }).fillna(0)

        pastel_bar_chart(comp, "Platform Comparison")

    # ---------- CHANNEL ----------
    elif input_type == "YouTube Channel Insights":
        cid = get_channel_id(channel_name)
        if not cid:
            st.error("Channel not found")
        else:
            stats = get_channel_stats(cid)
            df = analyze_channel_sentiment(cid)

            st.metric("Subscribers", stats["Subscribers"])
            st.metric("Total Views", stats["Total Views"])
            st.metric("Total Videos", stats["Total Videos"])

            pastel_pie_chart(df["Sentiment"].value_counts(), "Audience Sentiment")

st.markdown("---")
st.caption("Sentiment Analysis Studio | Real-Time Media Opinion Analysis Using ML")
