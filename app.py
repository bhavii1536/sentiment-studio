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
    def predict_sentiment(text):
        try:
            label = sentiment_model(text[:512])[0]["label"]
            return "Positive" if label == "POSITIVE" else "Negative"
        except:
            return "Negative"

    ASPECTS = {
        "Price": ["price", "cost", "expensive", "cheap"],
        "Quality": ["quality", "pure", "organic"],
        "Packaging": ["packaging", "bottle", "box"],
        "Delivery": ["delivery", "shipping"]
    }

    def aspect_sentiment(texts):
        rows = []
        for t in texts:
            tl = t.lower()
            for asp, keys in ASPECTS.items():
                if any(k in tl for k in keys):
                    rows.append({
                        "Aspect": asp,
                        "Sentiment": predict_sentiment(t)
                    })
        return pd.DataFrame(rows)

    def pie_bar(sentiments, small=True):
        s = pd.Series(sentiments).value_counts()

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            ax.pie(
                s,
                labels=s.index,
                autopct="%1.1f%%",
                colors=COLORS,
                startangle=90
            )
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(4, 3))
            s.plot(kind="bar", color=COLORS, ax=ax)
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Comparison")
            st.pyplot(fig)

    # ===============================
    # MODE SELECT
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
    # 1️⃣ PRODUCT / TOPIC ANALYSIS
    # ===============================
    if mode == "Product / Topic Analysis (YouTube)":
        topic = st.text_input("Enter product / topic")

        if st.button("Analyze"):
            st.info(f"🔍 Analyzing public opinion on: {topic}")

            videos = youtube.search().list(
                q=topic, part="id", type="video", maxResults=10
            ).execute()["items"]

            comments = []
            for v in videos:
                vid = v["id"]["videoId"]
                try:
                    res = youtube.commentThreads().list(
                        part="snippet", videoId=vid, maxResults=100
                    ).execute()
                    comments.extend([
                        i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                        for i in res["items"]
                    ])
                except:
                    pass

            if not comments:
                st.warning("No comments found.")
                st.stop()

            sentiments = [predict_sentiment(c) for c in comments]

            pie_bar(sentiments)

            st.subheader("📄 Sample Comments")
            for i, c in enumerate(comments[:5], 1):
                st.write(f"{i}. {c}")

            absa = aspect_sentiment(comments)
            if not absa.empty:
                st.subheader("🧠 Aspect-Based Sentiment")
                st.bar_chart(absa.value_counts().unstack().fillna(0))

    # ===============================
    # 2️⃣ CHANNEL INSIGHTS
    # ===============================
    elif mode == "YouTube Channel Insights":
        channel_name = st.text_input("Enter Channel Name")

        if st.button("Analyze Channel"):
            res = youtube.search().list(
                q=channel_name, part="snippet", type="channel", maxResults=1
            ).execute()

            if not res["items"]:
                st.error("Channel not found")
                st.stop()

            cid = res["items"][0]["snippet"]["channelId"]

            videos = youtube.search().list(
                channelId=cid, part="id", type="video", maxResults=25
            ).execute()["items"]

            total_views = total_likes = total_comments = 0
            views_per_video = []
            all_comments = []

            for v in videos:
                vid = v["id"]["videoId"]
                stats = youtube.videos().list(
                    part="statistics", id=vid
                ).execute()["items"][0]["statistics"]

                views = int(stats.get("viewCount", 0))
                likes = int(stats.get("likeCount", 0))
                total_views += views
                total_likes += likes
                views_per_video.append(views)

                try:
                    res = youtube.commentThreads().list(
                        part="snippet", videoId=vid, maxResults=50
                    ).execute()
                    total_comments += len(res["items"])
                    all_comments.extend([
                        i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                        for i in res["items"]
                    ])
                except:
                    pass

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Videos", len(videos))
            m2.metric("Comments", total_comments)
            m3.metric("Views", f"{total_views:,}")
            m4.metric("Likes", f"{total_likes:,}")

            # Weekly-style comparison (per video)
            st.subheader("📈 Views per Video (Recent)")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(range(len(views_per_video)), sorted(views_per_video), color="#60A5FA")
            ax.set_ylabel("Views")
            ax.set_xlabel("Videos")
            st.pyplot(fig)

            if all_comments:
                sentiments = [predict_sentiment(c) for c in all_comments]
                st.subheader("🙂 Audience Sentiment")
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                pd.Series(sentiments).value_counts().plot(
                    kind="pie",
                    autopct="%1.1f%%",
                    colors=COLORS,
                    ax=ax
                )
                ax.set_ylabel("")
                st.pyplot(fig)

    # ===============================
    # 3️⃣ CSV UPLOAD ANALYSIS
    # ===============================
    else:
        file = st.file_uploader("Upload CSV", type=["csv"])

        if file and st.button("Analyze Dataset"):
            try:
                df = pd.read_csv(file, encoding="utf-8")
            except:
                df = pd.read_csv(file, encoding="latin1")

            st.success(f"CSV loaded: {len(df)} rows")

            TEXT_COLS = ["text", "tweet", "review", "comment", "content"]
            text_col = None

            for c in df.columns:
                if c.lower() in TEXT_COLS:
                    text_col = c
                    break

            if not text_col:
                for c in df.columns:
                    if df[c].dtype == "object":
                        text_col = c
                        break

            if not text_col:
                st.error("No text column found")
                st.stop()

            st.success(f"Using column: {text_col}")

            texts = df[text_col].astype(str).tolist()[:2000]
            sentiments = [predict_sentiment(t) for t in texts]

            pie_bar(sentiments)

            st.subheader("📄 Sample Rows")
            for i, t in enumerate(texts[:5], 1):
                st.write(f"{i}. {t}")
