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

    # ===============================
    # COLORS
    # ===============================
    POS_COLOR = "#2563eb"   # Blue
    NEG_COLOR = "#f97316"   # Orange
    NEU_COLOR = "#9ca3af"   # Gray
    COLORS = [POS_COLOR, NEG_COLOR, NEU_COLOR]

    # ===============================
    # LOAD MODELS (CPU FRIENDLY)
    # ===============================
    @st.cache_resource
    def load_models():
        en = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        multi = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
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

    # ✅ FIXED SENTIMENT LOGIC (NO MORE ALWAYS NEGATIVE)
    def predict_sentiment(text):
        model = sentiment_en if detect_language(text) == "en" else sentiment_multi
        result = model(text[:512])[0]

        label = result["label"]
        score = result.get("score", 1.0)

        # Confidence-based neutral handling
        if score < 0.60:
            return "Neutral"

        if label in ["LABEL_1", "POSITIVE", "4 stars", "5 stars"]:
            return "Positive"
        elif label in ["LABEL_0", "NEGATIVE", "1 star", "2 stars"]:
            return "Negative"
        else:
            return "Neutral"

    # ===============================
    # ASPECTS (PRODUCT ONLY)
    # ===============================
    ASPECTS = {
        "Price": ["price", "cost", "expensive", "cheap"],
        "Quality": ["quality", "performance", "good", "bad"],
        "Battery": ["battery", "backup", "drain"],
        "Camera": ["camera", "photo", "video"],
    }

    def aspect_based_sentiment(texts):
        rows = []
        for t in texts:
            tl = t.lower()
            for a, keys in ASPECTS.items():
                if any(k in tl for k in keys):
                    rows.append({
                        "Aspect": a,
                        "Sentiment": predict_sentiment(t)
                    })
        return pd.DataFrame(rows)

    # ===============================
    # YOUTUBE FUNCTIONS
    # ===============================
    def search_videos(query, limit=10):
        res = youtube.search().list(
            q=query, part="id", type="video", maxResults=limit
        ).execute()
        return [i["id"]["videoId"] for i in res["items"]]

    def fetch_comments(video_id, limit=100):
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
                colors=[POS_COLOR, NEG_COLOR, NEU_COLOR][:len(s)]
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
    # TABS (UI FIXED)
    # ===============================
    tab1, tab2, tab3 = st.tabs([
        "📦 Product / Topic (YouTube)",
        "📊 Channel Insights",
        "📁 CSV Upload"
    ])

    # ======================================================
    # 📦 PRODUCT / TOPIC
    # ======================================================
    with tab1:
        topic = st.text_input("Enter product / topic")

        if st.button("Analyze Topic"):
            st.info(f"🔍 Analyzing public opinion on: {topic}")

            comments = []
            for vid in search_videos(topic, 10):
                comments.extend(fetch_comments(vid, 100))

            st.success(f"Fetched {len(comments)} comments")

            sentiments = [predict_sentiment(c) for c in comments]
            show_sentiment_charts(sentiments)

            st.subheader("📄 Sample Comments")
            for i, c in enumerate(comments[:5], 1):
                st.write(f"{i}. {c}")

            st.subheader("🧠 Aspect-Based Sentiment")
            absa = aspect_based_sentiment(comments)
            if not absa.empty:
                st.bar_chart(absa.value_counts().unstack().fillna(0))

    # ======================================================
    # 📊 CHANNEL INSIGHTS
    # ======================================================
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
                ).execute()

                views = []
                likes = 0
                comments = []

                for v in videos["items"]:
                    vid = v["id"]["videoId"]
                    stats = youtube.videos().list(
                        part="statistics", id=vid
                    ).execute()["items"][0]["statistics"]

                    views.append(int(stats.get("viewCount", 0)))
                    likes += int(stats.get("likeCount", 0))
                    comments.extend(fetch_comments(vid, 40))

                st.metric("Videos", len(videos["items"]))
                st.metric("Comments", len(comments))
                st.metric("Total Views", f"{sum(views):,}")
                st.metric("Total Likes", f"{likes:,}")

                # Weekly / per-video comparison
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(range(len(views)), views, color=POS_COLOR)
                ax.set_title("Views per Video (Recent Uploads)")
                ax.set_xlabel("Video Index")
                ax.set_ylabel("Views")
                st.pyplot(fig)

                sentiments = [predict_sentiment(c) for c in comments]
                show_sentiment_charts(sentiments)

    # ======================================================
    # 📁 CSV UPLOAD
    # ======================================================
    with tab3:
        file = st.file_uploader("Upload CSV", type="csv")

        if file:
            if st.button("Analyze Dataset"):
                try:
                    df = pd.read_csv(file, encoding="utf-8")
                except:
                    df = pd.read_csv(file, encoding="latin1")

                df.columns = (
                    df.columns.str.strip()
                    .str.lower()
                    .str.replace("\ufeff", "")
                )

                st.success(f"CSV loaded: {len(df)} rows")
                st.write("Detected columns:", list(df.columns))

                text_cols = [
                    "text", "tweet", "comment",
                    "content", "review", "sentence"
                ]

                text_col = next((c for c in text_cols if c in df.columns), None)

                if not text_col:
                    st.error("No usable text column found.")
                else:
                    texts = df[text_col].astype(str).head(1000)
                    sentiments = texts.apply(predict_sentiment)
                    show_sentiment_charts(sentiments)
