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

    POS_COLOR = "#F97316"   # Orange
    NEG_COLOR = "#2563EB"   # Blue
    NEU_COLOR = "#14B8A6"   # Teal
    COLORS = [POS_COLOR, NEG_COLOR, NEU_COLOR]

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
    youtube = build("youtube", "v3", developerKey=st.secrets["YOUTUBE_API_KEY"])

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
        res = model(text[:512])[0]
        label = res["label"]
        score = res["score"]

        if score < 0.60:
            return "Neutral"
        if label in ["LABEL_1", "POSITIVE"]:
            return "Positive"
        return "Negative"

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

    # ===============================
    # YOUTUBE FUNCTIONS
    # ===============================
    def search_videos(query, limit=15):
        res = youtube.search().list(
            q=query, part="id", type="video", maxResults=limit
        ).execute()
        return [i["id"]["videoId"] for i in res["items"]]

    def fetch_comments(video_id, limit=120):
        res = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100
        ).execute()
        return [
            i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            for i in res["items"][:limit]
        ]

    # ===============================
    # CHARTS
    # ===============================
    def show_sentiment_charts(sentiments):
        s = pd.Series(sentiments).value_counts()

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            ax.pie(
                s,
                labels=s.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=[POS_COLOR, NEG_COLOR, NEU_COLOR]
            )
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(3.5, 3))
            s.plot(kind="bar", color=[POS_COLOR, NEG_COLOR, NEU_COLOR], ax=ax)
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
    # PRODUCT / TOPIC
    # ===============================
    if mode == "Product / Topic Analysis (YouTube)":
        topic = st.text_input("Enter product / topic")

        if st.button("Analyze"):
            st.info(f"🔍 Analyzing public opinion on: {topic}")

            comments = []
            for vid in search_videos(topic):
                comments.extend(fetch_comments(vid))

            st.success(f"Analyzed {len(comments)} comments")

            sentiments = [predict_sentiment(c) for c in comments]
            show_sentiment_charts(sentiments)

            st.subheader("📄 Sample Comments")
            for i, c in enumerate(comments[:5], 1):
                st.write(f"{i}. {c}")

            st.subheader("🧠 Aspect-Based Sentiment")
            absa = aspect_based_sentiment(comments)
            if not absa.empty:
                st.bar_chart(absa.value_counts().unstack().fillna(0))

    # ===============================
    # CHANNEL INSIGHTS
    # ===============================
    elif mode == "YouTube Channel Insights":
        channel = st.text_input("Enter Channel Name")

        if st.button("Analyze Channel"):
            search = youtube.search().list(
                q=channel, part="snippet", type="channel", maxResults=1
            ).execute()

            if search["items"]:
                cid = search["items"][0]["snippet"]["channelId"]

                videos = youtube.search().list(
                    channelId=cid, part="id", type="video", maxResults=25
                ).execute()

                views = []
                likes = 0
                comments = []

                for v in videos["items"]:
                    vid = v["id"]["videoId"]
                    data = youtube.videos().list(
                        part="statistics", id=vid
                    ).execute()["items"][0]["statistics"]

                    views.append(int(data.get("viewCount", 0)))
                    likes += int(data.get("likeCount", 0))
                    comments.extend(fetch_comments(vid, 50))

                st.metric("Videos Analyzed", len(videos["items"]))
                st.metric("Total Comments", len(comments))
                st.metric("Total Views", f"{sum(views):,}")
                st.metric("Total Likes", f"{likes:,}")

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(range(len(views)), sorted(views), color=POS_COLOR)
                ax.set_title("Views per Video (Recent)")
                st.pyplot(fig)

                sentiments = [predict_sentiment(c) for c in comments]
                show_sentiment_charts(sentiments)

    # ===============================
    # CSV UPLOAD
    # ===============================
    else:
        file = st.file_uploader("Upload CSV", type="csv")

        if file:
            if st.button("Analyze Dataset"):
                try:
                    df = pd.read_csv(file, encoding="utf-8")
                except:
                    df = pd.read_csv(file, encoding="latin1")

                df.columns = df.columns.str.lower().str.strip()

                text_cols = ["text", "tweet", "comment", "review", "content", "sentence"]
                col = next((c for c in text_cols if c in df.columns), None)

                if not col:
                    st.error("No text column found")
                else:
                    texts = df[col].astype(str).head(2000)
                    sentiments = texts.apply(predict_sentiment)
                    show_sentiment_charts(sentiments)
