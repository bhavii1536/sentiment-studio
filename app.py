import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
from transformers import pipeline
from googleapiclient.discovery import build

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Sentiment Analysis Studio", layout="wide")

# ===============================
# STYLE
# ===============================
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 14px; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ===============================
# CENTER LAYOUT
# ===============================
_, main_col, _ = st.columns([1, 4, 1])

with main_col:

    st.title("📊 Sentiment Analysis Studio")
    st.caption("Real-Time Media Opinion Analysis Using Machine Learning")

    # COLORS
    POS = "#f97316"     # Orange
    NEU = "#2563eb"     # Blue
    NEG = "#ef4444"     # Red
    COLORS = [NEG, NEU, POS]

    # ===============================
    # LOAD MODELS
    # ===============================
    @st.cache_resource
    def load_model():
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    model = load_model()

    # ===============================
    # YOUTUBE API
    # ===============================
    youtube = build("youtube", "v3", developerKey=st.secrets["YOUTUBE_API_KEY"])

    # ===============================
    # HELPERS
    # ===============================
    def predict_sentiment(text):
        try:
            label = model(text[:512])[0]["label"]
            return "Positive" if label == "POSITIVE" else "Negative"
        except:
            return "Neutral"

    PRODUCT_ASPECTS = {
        "Price": ["price", "cost", "expensive", "cheap"],
        "Quality": ["quality", "performance", "build"],
        "Camera": ["camera", "photo", "video"],
        "Battery": ["battery", "charge", "backup"]
    }

    def aspect_based_sentiment(texts):
        rows = []
        for t in texts:
            tl = t.lower()
            for asp, keys in PRODUCT_ASPECTS.items():
                if any(k in tl for k in keys):
                    rows.append({"Aspect": asp, "Sentiment": predict_sentiment(t)})
        return pd.DataFrame(rows)

    def search_videos(query, limit=10):
        res = youtube.search().list(
            q=query, part="id", type="video", maxResults=limit
        ).execute()
        return [i["id"]["videoId"] for i in res["items"]]

    def fetch_comments(video_id, limit=100):
        try:
            res = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=limit
            ).execute()
            return [
                i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for i in res["items"]
            ]
        except:
            return []

    # ===============================
    # CHARTS
    # ===============================
    def sentiment_charts(sentiments):
        s = pd.Series(sentiments).value_counts()

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            ax.pie(s, labels=s.index, autopct="%1.1f%%", colors=COLORS, startangle=90)
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(3.6, 3))
            s.plot(kind="bar", color=COLORS, ax=ax)
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Comparison")
            st.pyplot(fig)

    # ===============================
    # MODE SELECT (TABS)
    # ===============================
    tab1, tab2, tab3 = st.tabs(
        ["📦 Product / Topic (YouTube)", "📺 Channel Insights", "📁 CSV Upload"]
    )

    # ===============================
    # PRODUCT / TOPIC
    # ===============================
    with tab1:
        analysis_type = st.radio(
            "What are you analyzing?",
            ["Product", "General Topic (Song / Movie / News)"]
        )

        topic = st.text_input("Enter product / topic")

        if st.button("Analyze Topic"):
            st.info(f"🔍 Analyzing public opinion on: {topic}")

            comments = []
            for vid in search_videos(topic):
                comments.extend(fetch_comments(vid))

            st.success(f"Fetched {len(comments)} comments")

            sentiments = [predict_sentiment(c) for c in comments]
            sentiment_charts(sentiments)

            st.subheader("📄 Sample Comments")
            for i, c in enumerate(comments[:5], 1):
                st.write(f"{i}. {c}")

            if analysis_type == "Product":
                st.subheader("🧠 Aspect-Based Sentiment")
                absa = aspect_based_sentiment(comments)
                if not absa.empty:
                    st.bar_chart(absa.value_counts().unstack().fillna(0))
            else:
                st.info(
                    "Aspect-based sentiment is not applicable for songs, movies, or general topics."
                )

    # ===============================
    # CHANNEL INSIGHTS
    # ===============================
    with tab2:
        channel = st.text_input("Enter Channel Name")

        if st.button("Analyze Channel"):
            search = youtube.search().list(
                q=channel, part="snippet", type="channel", maxResults=1
            ).execute()

            if search["items"]:
                cid = search["items"][0]["snippet"]["channelId"]

                videos = youtube.search().list(
                    channelId=cid, part="id", type="video", maxResults=25
                ).execute()["items"]

                views, likes, comments = [], 0, []

                for v in videos:
                    vid = v["id"]["videoId"]
                    stats = youtube.videos().list(
                        part="statistics", id=vid
                    ).execute()["items"][0]["statistics"]

                    views.append(int(stats.get("viewCount", 0)))
                    likes += int(stats.get("likeCount", 0))
                    comments.extend(fetch_comments(vid, 40))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Videos", len(videos))
                c2.metric("Comments", len(comments))
                c3.metric("Total Views", f"{sum(views):,}")
                c4.metric("Total Likes", f"{likes:,}")

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(range(len(views)), sorted(views, reverse=True), color=POS)
                ax.set_title("Views per Video (Recent)")
                ax.set_ylabel("Views")
                st.pyplot(fig)

                sentiments = [predict_sentiment(c) for c in comments]
                sentiment_charts(sentiments)

    # ===============================
    # CSV UPLOAD
    # ===============================
    with tab3:
        file = st.file_uploader("Upload CSV", type="csv")

        if file and st.button("Analyze Dataset"):
            try:
                df = pd.read_csv(file, encoding="utf-8")
            except:
                df = pd.read_csv(file, encoding="latin1")

            df.columns = df.columns.str.lower().str.strip()

            st.success(f"CSV loaded: {len(df)} rows")
            st.write("Detected columns:", list(df.columns))

            text_cols = ["text", "tweet", "comment", "review", "content"]
            col = next((c for c in text_cols if c in df.columns), None)

            if not col:
                st.error("No text column found")
            else:
                texts = df[col].astype(str).head(1000)
                sentiments = texts.apply(predict_sentiment)
                sentiment_charts(sentiments)
