import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
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
# CSS (DARK, CLEAN)
# ===============================
st.markdown("""
<style>
html, body {
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

    PIE_COLORS = ["#2563EB", "#F97316"]  # blue, orange

    # ===============================
    # LOAD MODEL (FAST + MULTILINGUAL)
    # ===============================
    @st.cache_resource
    def load_model():
        return pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
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
    # SENTIMENT PREDICTION
    # ===============================
    def predict_sentiment(text):
        try:
            label = sentiment_model(text[:512])[0]["label"]
            stars = int(label[0])
            return "Negative" if stars <= 2 else "Positive"
        except:
            return "Positive"

    # ===============================
    # ASPECT-BASED SENTIMENT
    # ===============================
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
            for aspect, keys in ASPECTS.items():
                if any(k in tl for k in keys):
                    rows.append({
                        "Aspect": aspect,
                        "Sentiment": predict_sentiment(t)
                    })
        return pd.DataFrame(rows)

    # ===============================
    # YOUTUBE HELPERS
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

    def fetch_video_stats(video_ids):
        res = youtube.videos().list(
            part="statistics,snippet",
            id=",".join(video_ids)
        ).execute()
        return res["items"]

    # ===============================
    # CHARTS
    # ===============================
    def sentiment_pie_bar(sentiments):
        s = pd.Series(sentiments).value_counts()

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ax.pie(
                s,
                labels=s.index,
                autopct="%1.1f%%",
                colors=PIE_COLORS,
                startangle=90
            )
            ax.axis("equal")
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(4, 3))
            s.plot(kind="bar", color=PIE_COLORS, ax=ax)
            ax.set_ylabel("Comments")
            ax.set_title("Sentiment Comparison")
            st.pyplot(fig)

    def aspect_bar(absa_df):
        chart = absa_df.value_counts().unstack().fillna(0)
        st.bar_chart(chart)

    def aspect_pies(absa_df):
        aspects = absa_df["Aspect"].unique()
        cols = st.columns(len(aspects))

        for col, asp in zip(cols, aspects):
            with col:
                data = (
                    absa_df[absa_df["Aspect"] == asp]["Sentiment"]
                    .value_counts()
                    .reindex(["Positive", "Negative"], fill_value=0)
                )
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.pie(
                    data,
                    labels=data.index,
                    autopct="%1.1f%%",
                    colors=["#22C55E", "#EF4444"],
                    startangle=90
                )
                ax.axis("equal")
                ax.set_title(asp)
                st.pyplot(fig)

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
    # PRODUCT / TOPIC
    # ===============================
    if mode == "Product / Topic Analysis (YouTube)":
        topic = st.text_input("Enter product / topic")

        if st.button("Analyze"):
            st.info(f"🔍 Analyzing public opinion on: {topic}")

            videos = search_videos(topic)
            comments = []
            for v in videos:
                comments.extend(fetch_comments(v))

            comments = comments[:800]

            st.success(f"Fetched {len(videos)} videos & {len(comments)} comments")

            sentiments = [predict_sentiment(c) for c in comments]
            sentiment_pie_bar(sentiments)

            st.subheader("🧠 Aspect-Based Sentiment")
            absa = aspect_based_sentiment(comments)
            if not absa.empty:
                aspect_bar(absa)

    # ===============================
    # CHANNEL INSIGHTS
    # ===============================
    elif mode == "YouTube Channel Insights":
        channel = st.text_input("Enter Channel Name")

        if st.button("Analyze Channel"):
            res = youtube.search().list(
                q=channel, part="snippet", type="channel", maxResults=1
            ).execute()

            if not res["items"]:
                st.error("Channel not found")
            else:
                cid = res["items"][0]["snippet"]["channelId"]

                vids = youtube.search().list(
                    channelId=cid, part="id", type="video", maxResults=12
                ).execute()

                video_ids = [v["id"]["videoId"] for v in vids["items"]]
                stats = fetch_video_stats(video_ids)

                total_views = 0
                total_likes = 0
                monthly = {}

                comments = []

                for v in stats:
                    s = v["statistics"]
                    snip = v["snippet"]
                    total_views += int(s.get("viewCount", 0))
                    total_likes += int(s.get("likeCount", 0))

                    date = snip["publishedAt"][:7]
                    monthly[date] = monthly.get(date, 0) + int(s.get("viewCount", 0))

                    comments.extend(fetch_comments(v["id"]))

                comments = comments[:800]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Videos", len(video_ids))
                m2.metric("Comments", len(comments))
                m3.metric("Total Views", f"{total_views:,}")
                m4.metric("Total Likes", f"{total_likes:,}")

                # LAST 6 MONTHS ORDERED
                now = datetime.now()
                months = [
                    (now.replace(day=1) - pd.DateOffset(months=i)).strftime("%Y-%m")
                    for i in range(5, -1, -1)
                ]

                views = [monthly.get(m, 0) for m in months]

                st.subheader("📈 Monthly Views (Last 6 Months)")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(months, views, color="#93C5FD")
                ax.set_ylabel("Views")
                st.pyplot(fig)

                sentiments = [predict_sentiment(c) for c in comments]
                sentiment_pie_bar(sentiments)

                absa = aspect_based_sentiment(comments)
                if not absa.empty:
                    st.subheader("🧠 Aspect-Based Sentiment")
                    aspect_pies(absa)

    # ===============================
    # CSV UPLOAD
    # ===============================
    else:
        file = st.file_uploader("Upload CSV (text column required)", type="csv")
        if file:
            df = pd.read_csv(file)
            if "text" not in df.columns:
                st.error("CSV must contain a 'text' column")
            else:
                sentiments = df["text"].apply(predict_sentiment)
                sentiment_pie_bar(sentiments)
